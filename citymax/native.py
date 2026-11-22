"""Document API access; model output cannot contain arbitrary key-ins or code."""
import json
import math
from pathlib import Path


def ensure_default_desktop():
    try:
        import win32service, win32con, ctypes
        h_desk = win32service.OpenDesktop('Default', 0, False, win32con.GENERIC_ALL)
        ctypes.windll.user32.SetThreadDesktop(int(h_desk))
    except Exception:
        pass

ensure_default_desktop()


def app():
    ensure_default_desktop()
    import comtypes.client
    return comtypes.client.GetActiveObject('MicroStationDGN.Application')


def snapshot(application=None):
    import comtypes.client
    application=application or app();file=Path(application.ActiveDesignFile.FullName)
    model=application.ActiveModelReference
    rasters=[]
    for i,r in enumerate(application.RasterManager.Rasters,1):
        r=comtypes.client.GetBestInterface(r)
        info=comtypes.client.GetBestInterface(r.RasterInformation)
        geo=comtypes.client.GetBestInterface(r.GeoReferenceInformation)
        tif=Path(info.FullName)
        if not tif.is_absolute():tif=file.parent/tif
        rasters.append({'index':i,'tif':str(tif.resolve()),'pixels':[info.Extent.X,info.Extent.Y],
            'origin':[geo.Origin.X,geo.Origin.Y],'extent':[geo.Extent.X,geo.Extent.Y],
            'rotation':geo.RotationAboutZ,'affinity':geo.Affinity})
    elements=[];scan=model.Scan()
    while scan.MoveNext():
        e=scan.Current
        if not e.IsGraphical:continue
        row={'id':int(e.ID64),'type':int(e.Type),'level':str(e.Level.Name),'level_number':int(e.Level.Number)}
        if e.IsVertexList:row['points']=[[p.X,p.Y,p.Z] for p in e.AsVertexList.GetVertices()]
        elif int(e.Type)==37:
            tag=e.AsTagElement
            row.update(tag=str(tag.TagDefinitionName),value=tag.Value,base=int(tag.BaseElement.ID64))
        elements.append(row)
    return {'dgn':str(file),'model_name':str(model.Name),
            'unit':str(model.MasterUnit.Label),
            'rasters':rasters,'elements':elements}


class RasterMap:
    def __init__(self,data):self.data=data

    def xy(self,x,y):
        d=self.data;u=x/d['pixels'][0]*d['extent'][0];v=(1-y/d['pixels'][1])*d['extent'][1]
        a=d['rotation'];b=a+d['affinity']
        return [d['origin'][0]+math.cos(a)*u-math.sin(b)*v,
                d['origin'][1]+math.sin(a)*u+math.cos(b)*v]

    def pixels(self,x,y):
        d=self.data;a=d['rotation'];b=a+d['affinity'];det=math.cos(a)*math.cos(b)+math.sin(a)*math.sin(b)
        if abs(det)<1e-8:raise ValueError('Degenerate raster transform')
        dx,dy=x-d['origin'][0],y-d['origin'][1]
        u=(math.cos(b)*dx+math.sin(b)*dy)/det
        v=(-math.sin(a)*dx+math.cos(a)*dy)/det
        return [u/d['extent'][0]*d['pixels'][0],(1-v/d['extent'][1])*d['pixels'][1]]


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    import os,tempfile
    handle=tempfile.NamedTemporaryFile(mode='w',encoding='utf8',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False)
    temporary=Path(handle.name)
    try:
        with handle:
            json.dump(data,handle,ensure_ascii=False,indent=2);handle.flush();os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()


def accept_owned_room_deletions(initial, current, owned_ids):
    """Accept only complete removals of known test rooms, never other edits.

    Returns the live baseline; it does not recreate or mutate any element.
    A newly authorized model proposal may then be written with a fresh receipt.
    """
    if any(initial[k]!=current[k] for k in ('dgn','unit','rasters')):
        raise RuntimeError('Drawing identity, units or raster registration changed')
    before={e['id']:e for e in initial['elements']};after={e['id']:e for e in current['elements']}
    if set(after)-set(before):raise RuntimeError('Other new elements appeared; preserve user work')
    missing=set(before)-set(after);allowed=set()
    for identity in owned_ids:
        if identity not in missing:continue
        if before[identity].get('level')!='OdaAlani':raise RuntimeError('Owned identity is no longer a test room')
        group={identity}|{e['id'] for e in initial['elements'] if e.get('base')==identity}
        if not group<=missing:raise RuntimeError('Partially removed room attributes; preserve user work')
        allowed|=group
    if missing-allowed:raise RuntimeError('Unrelated elements were removed; preserve user work')
    if any(after[k]!=before[k] for k in after):raise RuntimeError('Existing geometry or attributes changed; preserve user work')
    return current
