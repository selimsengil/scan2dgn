"""Fixed native geometry/Tag bridge. Never execute model-provided code or key-ins."""
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

from shapely.geometry import Polygon,LineString
from .native import app, snapshot, RasterMap, save

from .native_schema import LEVELS, OPEN_KINDS, TAGS
from .command_contract import prepare_command, explicit_attributes, canonical_points, validate_transport



def quote(value):return '"'+str(value).replace('"','""')+'"'


def source_raster(command, snapshot):
    """Resolve only the caller's explicit raster identity; never choose a floor."""
    index=command.get('source_raster_index',1)
    if isinstance(index,bool) or not isinstance(index,int) or not 1<=index<=len(snapshot['rasters']):
        raise ValueError('Invalid explicit source raster index')
    return snapshot['rasters'][index-1]


def native_attributes(command,live=None):
    """Compatibility API: final native values only; live geometry is not read."""
    return explicit_attributes(command)


def correct_attributes(command,initial,directory,reason):
    """Only revise this writer's unchanged component; keep its native IDs."""
    command=prepare_command(command,initial)
    directory=Path(directory);receipt=directory/(command['key']+'_native.json')
    prior=json.loads(receipt.read_text(encoding='utf8'));application=app();live=snapshot(application)
    if Path(prior['dgn']).resolve()!=Path(live['dgn']).resolve():raise RuntimeError('Receipt belongs to another DGN')
    if live['dgn']!=initial['dgn'] or live['rasters']!=initial['rasters']:raise RuntimeError('Correction source changed')
    if live['unit']!='m':raise RuntimeError('Correction requires metre units')
    if Path(command.get('source_dgn',live['dgn'])).resolve()!=Path(live['dgn']).resolve():raise RuntimeError('Correction belongs to another DGN')
    if command.get('source_sha256'):
        with Path(source_raster(command,live)['tif']).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=command['source_sha256']:raise RuntimeError('Correction source pixels changed')
    existing=next(e for e in live['elements'] if e['id']==prior['id'])
    linked=[e for e in live['elements'] if e.get('base')==prior['id']]
    if existing['points']!=prior['native_points'] or linked!=prior['tags']:raise RuntimeError('Preserve user-edited shape/attributes')
    kind=command['kind'];definitions=TAGS[kind][1];attrs=native_attributes(command,live)
    if existing['level']!=LEVELS[kind][0] or set(attrs)!={d[0] for d in definitions}:raise ValueError('Wrong kind or incomplete attributes')
    mapper=RasterMap(source_raster(command,live));expected=[mapper.xy(*p) for p in command['vertices_raw']]
    actual=canonical_points([p[:2] for p in existing['points']],closed=kind not in OPEN_KINDS)
    if len(actual)!=len(expected) or any(math.dist(p[:2],q)>1e-8 for p,q in zip(actual,expected)):raise ValueError('Attribute correction cannot change geometry')
    for value in attrs.values():value_vba(value)
    application.ActiveDesignFile.Save()
    backup=directory/(command['key']+'_before_attribute_correction.dgn')
    if not backup.exists():shutil.copy2(live['dgn'],backup)
    save(directory/(command['key']+'_attribute_correction.json'),{'before':prior,'command':command,'reason':reason,'status':'prepared'})
    try:
        for row in linked:
            element=application.ActiveModelReference.GetElementByID64(row['id']);tag=element.AsTagElement
            value=attrs[row['tag']];tag.Value=-1.0 if value is None else value;tag.IsHidden=value is None;element.Rewrite()
        application.ActiveDesignFile.Save();after=snapshot(application)
        new_tags=[e for e in after['elements'] if e.get('base')==prior['id']]
        if {t['tag']:t['value'] for t in new_tags}!={k:(-1.0 if v is None else v) for k,v in attrs.items()}:raise RuntimeError('Attribute readback mismatch')
    except Exception:
        for row in linked:
            element=application.ActiveModelReference.GetElementByID64(row['id']);tag=element.AsTagElement
            tag.Value=row['value'];tag.IsHidden=row['value']==-1.0;element.Rewrite()
        application.ActiveDesignFile.Save();raise
    result={**prior,'tags':new_tags,'digest':hashlib.sha256(json.dumps(command,sort_keys=True).encode()).hexdigest(),'attribute_correction_reason':reason}
    save(receipt,result);save(directory/(command['key']+'_attribute_correction.json'),{'before':prior,'after':result,'command':command,'reason':reason,'status':'verified'})
    return result


def value_vba(value):
    if value is None:return '-1#'
    if isinstance(value,str):return quote(value)
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Invalid value')
    return str(value).replace(',','.')


def correct_geometry(command,initial,directory,reason):
    """Explicit correction of this writer's own unchanged shape, retaining Tag IDs.

    It cannot modify pre-existing/user geometry: a matching identity receipt is
    required. Vertex changes are prepared in memory and rewritten only once.
    """
    command=prepare_command(command,initial)
    directory=Path(directory).resolve(); receipt=directory/f'{command["key"]}_native.json'
    prior=json.loads(receipt.read_text(encoding='utf8'))
    application=app();live=snapshot(application)
    if Path(prior['dgn']).resolve()!=Path(live['dgn']).resolve():raise RuntimeError('Receipt belongs to another DGN')
    if live['dgn']!=initial['dgn'] or live['rasters']!=initial['rasters']:raise RuntimeError('Correction source changed')
    if live['unit']!='m':raise RuntimeError('Correction requires metre units')
    if command.get('source_dgn') and Path(command['source_dgn']).resolve()!=Path(live['dgn']).resolve():raise RuntimeError('Correction belongs to another DGN')
    if command.get('source_sha256'):
        with Path(source_raster(command,live)['tif']).open('rb') as stream:actual_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual_sha!=command['source_sha256']:raise RuntimeError('Correction source pixels changed')
    existing=next(e for e in live['elements'] if e['id']==prior['id'])
    if command['kind'] not in LEVELS or (command['kind'] in OPEN_KINDS and command['kind']!='bb_boundary') or existing['level']!=LEVELS[command['kind']][0]:raise ValueError('Correction kind does not match original shape')
    if set(command['tags'])!={r[0] for r in TAGS.get(command['kind'],('',[]))[1]}:raise ValueError('Incomplete correction attributes')
    raster=source_raster(command,live)
    if any(not 0<=p[0]<=raster['pixels'][0] or not 0<=p[1]<=raster['pixels'][1] for p in command['vertices_raw']):raise ValueError('Correction outside source image')
    if existing['points']!=prior['native_points']:raise RuntimeError('Shape changed since receipt; preserve user edit')
    linked=[e for e in live['elements'] if e.get('base')==prior['id']]
    if linked!=prior['tags']:raise RuntimeError('Linked attributes changed; preserve user edit')
    label_positions={}
    for row in linked:
        tag=application.ActiveModelReference.GetElementByID64(row['id']).AsTagElement
        origin,offset=tag.Origin,tag.Offset
        label_positions[row['id']]=[origin.X+offset.X,origin.Y+offset.Y,origin.Z+offset.Z]
    if {t['tag']:t['value'] for t in linked}!={k:(-1.0 if v is None else v) for k,v in native_attributes(command,live).items()}:raise ValueError('Geometry correction cannot change attributes')
    application.ActiveDesignFile.Save()
    history=directory/(command['key']+'_correction_history.json')
    rows=json.loads(history.read_text(encoding='utf8')) if history.exists() else []
    rows.append({'before':prior,'command':command,'reason':reason,'status':'prepared'})
    save(history,rows)
    backup=directory/(command['key']+'_before_correction.dgn')
    if not backup.exists():shutil.copy2(initial['dgn'],backup)
    mapping=RasterMap(source_raster(command,initial));points=[[round(v,12) for v in mapping.xy(*p)] for p in command['vertices_raw']]
    if command['kind'] not in OPEN_KINDS:points=points+[points[0]]
    element=application.ActiveModelReference.GetElementByID64(prior['id']); vertices=element.AsVertexList
    while vertices.VerticesCount>len(points):vertices.RemoveVertex(1)
    while vertices.VerticesCount<len(points):vertices.InsertVertex(1,vertices.GetVertices()[0])
    for i,(x,y) in enumerate(points):vertices.ModifyVertex(i,application.Point3dFromXYZ(x,y,0))
    element.Rewrite();application.CommandState.StartDefaultCommand()
    for eid,target in label_positions.items():
        tag_element=application.ActiveModelReference.GetElementByID64(eid);tag=tag_element.AsTagElement
        origin,offset=tag.Origin,tag.Offset
        delta=[target[0]-origin.X-offset.X,target[1]-origin.Y-offset.Y,target[2]-origin.Z-offset.Z]
        if math.dist(delta,[0,0,0])>1e-8:tag_element.Move(application.Point3dFromXYZ(*delta));tag_element.Rewrite()
    application.ActiveDesignFile.Save();application.ActiveDesignFile.Views(1).Redraw()
    after=snapshot(application);actual=next(e for e in after['elements'] if e['id']==prior['id'])
    if len(actual['points'])!=len(points) or any(math.dist(q[:2],p)>1e-8 for q,p in zip(actual['points'],points)):raise RuntimeError('Correction readback mismatch')
    if [e for e in after['elements'] if e.get('base')==prior['id']]!=linked:raise RuntimeError('Correction changed linked Tags')
    result={**prior,'digest':hashlib.sha256(json.dumps(command,sort_keys=True).encode()).hexdigest(),'native_points':actual['points'],'correction_reason':reason}
    save(receipt,result);rows[-1]['status']='verified';save(history,rows);return result


def write_command(command, initial, directory):
    """One component transaction with pre-write backup and durable identity receipt."""
    import comtypes.client
    application=app();live=snapshot(application)
    if Path(live['dgn']).resolve()!=Path(initial['dgn']).resolve():raise RuntimeError('Active DGN changed')
    if command.get('source_dgn') and Path(command['source_dgn']).resolve()!=Path(live['dgn']).resolve():raise RuntimeError('Command belongs to another source DGN')
    if live['unit']!='m' or live['rasters']!=initial['rasters']:raise RuntimeError('Raster transform or units changed')
    if command.get('source_sha256'):
        with Path(source_raster(command,live)['tif']).open('rb') as stream:actual_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual_sha!=command['source_sha256']:raise RuntimeError('Source pixels changed')
    transport=validate_transport(command,initial)
    command=prepare_command(command,initial)
    kind=command['kind'];level,number,color=LEVELS[kind];points=command['vertices_raw']
    polygon=LineString(points) if kind in OPEN_KINDS else Polygon(points)
    attrs=native_attributes(command)
    key=command['key']
    if not key.replace('_','').isalnum() or len(key)>80:raise ValueError('Invalid identity')
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/f'{key}_native.json';digest=hashlib.sha256(json.dumps(command,sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        prior=json.loads(receipt.read_text(encoding='utf8'))
        if Path(prior['dgn']).resolve()!=Path(live['dgn']).resolve():raise RuntimeError('Receipt belongs to another DGN')
        if prior['digest']!=digest:raise RuntimeError('Changed candidate needs explicit correction transaction')
        existing=next((e for e in live['elements'] if e['id']==prior['id']),None)
        attached=[e for e in live['elements'] if e.get('base')==prior['id']]
        if existing and existing['points']==prior['native_points'] and attached==prior['tags']:
            if {t['tag']:t['value'] for t in attached}!={k:(-1.0 if v is None else v) for k,v in attrs.items()}:raise RuntimeError('Native attribute representation changed; explicit correction required')
            return prior
        raise RuntimeError('Previously written geometry edited/removed; do not overwrite user work')
    application.ActiveDesignFile.Save()
    backup=directory/'before_first_write.dgn'
    if not backup.exists():shutil.copy2(initial['dgn'],backup)
    requirements=[(level,number)]+[(r[2],r[3]) for r in TAGS.get(kind,('',[]))[1]]
    known={lv.Name:lv for lv in application.ActiveDesignFile.Levels}
    numbers={lv.Number:lv.Name for lv in application.ActiveDesignFile.Levels}
    for name,code in requirements:
        if name in known:lv=known[name]
        else:
            if code in numbers:raise RuntimeError('Citymax level number occupied: '+str(code))
            lv=application.ActiveDesignFile.AddNewLevel(name);lv.Number=code
        if lv.Number!=code:raise RuntimeError('Unexpected Citymax level number: '+name)
    application.ActiveDesignFile.RewriteLevels()
    mapping=RasterMap(source_raster(command,initial));world=[mapping.xy(*p) for p in points]
    done=directory/f'{key}.id';error=directory/f'{key}.error'
    for f in (done,error):
        if f.exists():f.unlink()
    code=['Option Explicit','Public Sub WriteComponent()','On Error GoTo Failed',
        'Dim p() As Point3d, e As Element, t As TagElement, ts As TagSet, f As Integer, sty As TextStyle, oldStyle As TextStyle',
        f'If LCase$(ActiveDesignFile.FullName) <> LCase$({quote(initial["dgn"])}) Then Err.Raise 5',
        f'ReDim p(0 To {len(world)-1})']
    code.extend(f'p({i}) = Point3dFromXYZ({x:.12f},{y:.12f},0)' for i,(x,y) in enumerate(world))
    creation='CreateLineElement1(Nothing,p)' if kind in OPEN_KINDS else 'CreateShapeElement1(Nothing,p,msdFillModeNotFilled)'
    code.extend(['Set e = '+creation,f'Set e.Level = ActiveDesignFile.Levels({quote(level)})',f'e.Color = {color}: e.LineWeight = 1'])
    if kind == 'window':
        code.append('Set e.LineStyle = ActiveDesignFile.LineStyles("2")')
    code.append('ActiveModelReference.AddElement e')
    if kind in TAGS:
        tagset,definitions=TAGS[kind]
        code.extend(['Set ts = Nothing','On Error Resume Next',f'Set ts = ActiveDesignFile.TagSets({quote(tagset)})','On Error GoTo Failed',f'If ts Is Nothing Then Set ts = ActiveDesignFile.TagSets.Add({quote(tagset)})'])
        # Definition creation through document API; existing definitions remain intact.
        try:ts=application.ActiveDesignFile.TagSets.Item(tagset)
        except Exception:ts=application.ActiveDesignFile.TagSets.Add(tagset)
        for tag,tagtype,_,_ in definitions:
            try:ts.TagDefinitions.Item(tag)
            except Exception:ts.TagDefinitions.Add(tag,tagtype)
        code.extend(['Set oldStyle = ActiveSettings.TextStyle','Set sty = ActiveSettings.TextStyle','sty.Height = .20: sty.Width = .20','Set sty.Font = ActiveDesignFile.Fonts("Arial")','Set ActiveSettings.TextStyle = sty'])
        position=command.get('label_raw',list(polygon.representative_point().coords)[0]);x,y=mapping.xy(*position)
        for i,(tag,_,taglevel,_) in enumerate(definitions):
            code.extend([f'Set t = e.AddTag(ts.TagDefinitions({quote(tag)}))',f't.Value = {value_vba(attrs[tag])}',f'Set t.Level = ActiveDesignFile.Levels({quote(taglevel)})',f't.Move Point3dSubtract(Point3dFromXYZ({x:.12f},{y-i*.23:.12f},0),t.Origin)',f't.Color = {color}',f't.IsHidden = {"True" if attrs[tag] is None or attrs[tag] in ("","-1") else "False"}','t.Rewrite'])
        code.append('Set ActiveSettings.TextStyle = oldStyle')
    code.extend(['ActiveDesignFile.Save','ActiveDesignFile.Views(1).Redraw','f = FreeFile',f'Open {quote(done)} For Output As #f','Print #f, DLongToString(e.ID)','Close #f','Exit Sub','Failed:','f = FreeFile',f'Open {quote(error)} For Output As #f','Print #f, Err.Number & " " & Err.Description','Close #f','End Sub'])
    host=__import__('config').PROJECTS/'sinir_otomasyonu/SinirPilot.mvba'
    application.CadInputQueue.SendKeyin('vba load '+str(host))
    project=None
    for _ in range(30):
        try:
            vbe=comtypes.client.GetBestInterface(application.VBE)
            project=next((vbe.VBProjects.Item(i) for i in range(1,vbe.VBProjects.Count+1) if vbe.VBProjects.Item(i).Name=='SinirPilot'),None)
        except Exception:pass
        if project is not None:break
        time.sleep(.1)
    if project is None:raise RuntimeError('Native macro host could not be loaded')
    component=next((project.VBComponents.Item(i) for i in range(1,project.VBComponents.Count+1) if project.VBComponents.Item(i).Name=='ArchitectWrite'),None)
    if component is None:component=project.VBComponents.Add(1);component.Name='ArchitectWrite'
    module=component.CodeModule
    if module.CountOfLines:module.DeleteLines(1,module.CountOfLines)
    source='\r\n'.join(code);module.AddFromString(source);(directory/f'{key}.bas').write_text(source,encoding='utf8')
    application.CadInputQueue.SendKeyin('vba run [SinirPilot]ArchitectWrite.WriteComponent')
    for _ in range(100):
        if error.exists():raise RuntimeError(error.read_text(errors='replace'))
        if done.exists():break
        time.sleep(.1)
    else:raise RuntimeError('Native bridge did not complete')
    eid=int(done.read_text().strip());after=snapshot(application);element=next(e for e in after['elements'] if e['id']==eid)
    attached=[e for e in after['elements'] if e.get('base')==eid]
    if {t['tag']:t['value'] for t in attached}!={k:(-1.0 if v is None else v) for k,v in attrs.items()}:raise RuntimeError('Tag linkage/value mismatch')
    actual=canonical_points([p[:2] for p in element['points']],closed=kind not in OPEN_KINDS)
    if len(actual)!=len(world) or any(math.dist(p,q)>1e-8 for p,q in zip(actual,world)):
        raise RuntimeError('Native geometry differs from explicit command')
    result={'id':eid,'digest':digest,'dgn':initial['dgn'],'native_points':element['points'],'tags':attached,'command_attributes':command.get('tags',{}),'native_attributes':attrs,'status':'native_geometry_and_tags_verified','architectural_acceptance':'not_evaluated','command_contract':'pencil_v1',
        'transport':transport,'label_placement':'explicit' if command.get('label_raw') is not None else 'numeric_representative_point'}
    save(receipt,result);return result
