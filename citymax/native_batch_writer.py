"""Direct MicroStation document API transport for explicitly reviewed new shapes.

No generated VBA, source interpretation, snapping, or modification of existing
objects. The caller owns architectural decisions and review gates.
"""
from pathlib import Path
import hashlib,json,math,shutil
from shapely.geometry import Polygon,LineString
from .native import app,snapshot,RasterMap,save
from .native_schema import LEVELS,TAGS,OPEN_KINDS
from .command_contract import prepare_command
from .batch_writer import _require_drawing_identity,_require_native_format,_same_native_geometry

def _set_object_property(element,name,value):
    # The type library exposes both put and putref. comtypes selects putref,
    # which V8i rejects for these interfaces; call the documented put slot.
    setter=next(n for n in dir(element)if n.endswith('__com__set_'+name))
    getattr(element,setter)(value)

def _tag_state(model,eid):
    t=model.GetElementByID64(eid).AsTagElement;o,d=t.Origin,t.Offset
    return {'placement':[o.X+d.X,o.Y+d.Y,o.Z+d.Z],'font':str(t.Font.Name),
            'height':t.Height,'width':t.Width,'hidden':bool(t.IsHidden),'color':int(t.Color)}

def write_batch_com(commands,initial,directory,stage):
    if not isinstance(stage,str)or not stage.replace('_','').isalnum():raise ValueError('Invalid stage identity')
    a=app();before=snapshot(a);_require_drawing_identity(initial,before)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    prepared=[prepare_command(c,before)for c in commands]
    assert len({c['key'] for c in prepared})==len(prepared),'Duplicate command identity'
    rows=[];source_hashes={}
    for c in prepared:
        if c.get('replace_id') is not None:raise ValueError('New native components only')
        receipt=directory/(c['key']+'_native.json')
        if receipt.exists():raise RuntimeError('Existing receipt; inspect before repeating '+c['key'])
        m=RasterMap(before['rasters'][c.get('source_raster_index',1)-1]);world=[m.xy(*p)for p in c['vertices_raw']]
        if any(e.get('points')and e['level']==LEVELS[c['kind']][0]and _same_native_geometry(e['points'],world,closed=c['kind']not in OPEN_KINDS)for e in before['elements']):raise RuntimeError('Existing geometry: '+c['key'])
        if any(LEVELS[other['kind']][0]==LEVELS[c['kind']][0]and _same_native_geometry(prior_world,world,closed=c['kind']not in OPEN_KINDS)for other,_,prior_world,_ in rows):raise RuntimeError('Duplicate geometry in native batch')
        if c.get('source_sha256'):
            source_path=Path(m.data['tif'])
            if source_path not in source_hashes:
                with source_path.open('rb')as f:source_hashes[source_path]=hashlib.file_digest(f,'sha256').hexdigest()
            sha=source_hashes[source_path]
            if sha!=c['source_sha256']:raise RuntimeError('Source changed')
        rows.append((c,m,world,receipt))
    journal=directory/(stage+'_direct_transaction.json')
    if journal.exists():raise RuntimeError('Transaction exists; inspect before retry')
    existing_tags={e['id']:_tag_state(a.ActiveModelReference,e['id'])for e in before['elements']if e['type']==37}
    a.ActiveDesignFile.Save();backup=directory/'before_first_direct_batch.dgn'
    if not backup.exists():shutil.copy2(before['dgn'],backup)
    save(journal,{'status':'prepared','before_ids':[e['id']for e in before['elements']],'commands':prepared,'transport':'native_COM'})
    known={l.Name:l for l in a.ActiveDesignFile.Levels};occupied={l.Number:l.Name for l in a.ActiveDesignFile.Levels}
    for c,_,_,_ in rows:
        kind=c['kind'];defs=TAGS.get(kind,('',[]))[1]
        for name,number in [(LEVELS[kind][0],LEVELS[kind][1]),*[(r[2],r[3])for r in defs]]:
            if name not in known:
                if number in occupied:raise RuntimeError('Level number occupied')
                level=a.ActiveDesignFile.AddNewLevel(name);level.Number=number;known[name]=level
            if known[name].Number!=number:raise RuntimeError('Wrong native level')
        if kind in TAGS:
            name,defs=TAGS[kind]
            try:ts=a.ActiveDesignFile.TagSets.Item(name)
            except Exception:ts=a.ActiveDesignFile.TagSets.Add(name)
            for label,typ,_,_ in defs:
                try:td=ts.TagDefinitions.Item(label)
                except Exception:td=ts.TagDefinitions.Add(label,typ)
                if td.TagType!=typ:raise RuntimeError('Wrong native Tag definition')
    a.ActiveDesignFile.RewriteLevels();made=[];identities={};targets={};oldids={e['id']for e in before['elements']}
    try:
        for c,m,world,receipt in rows:
            kind=c['kind'];level,_,color=LEVELS[kind];points=[a.Point3dFromXYZ(*p,0)for p in world]
            result=a.CreateLineElement1(None,points)if kind in OPEN_KINDS else a.CreateShapeElement1(None,points,0)
            e=result[-1];_set_object_property(e,'Level',known[level]);e.Color=color;e.LineWeight=1
            if kind=='window':_set_object_property(e,'LineStyle',a.ActiveDesignFile.LineStyles.Item('2'))
            a.ActiveModelReference.AddElement(e);made.append(int(e.ID64));identities[c['key']]=int(e.ID64)
            if kind in TAGS:
                tagset,defs=TAGS[kind];ts=a.ActiveDesignFile.TagSets.Item(tagset)
                g=LineString(c['vertices_raw'])if kind in OPEN_KINDS else Polygon(c['vertices_raw'])
                position=c.get('label_raw',list(g.representative_point().coords)[0]);x,y=m.xy(*position)
                for ix,(label,_,taglevel,_) in enumerate(defs):
                    t=e.AddTag(ts.TagDefinitions.Item(label));made.append(int(t.ID64));value=c['tags'][label]
                    t.Value=-1.0 if value is None else value;_set_object_property(t,'Level',known[taglevel]);t.Height=.2;t.Width=.2
                    _set_object_property(t,'Font',a.ActiveDesignFile.Fonts.Item('Arial'));t.Color=color;t.IsHidden=value is None or value in ('','-1')
                    target=[x,y-ix*.23,0];origin=t.Origin;offset=t.Offset
                    t.Move(a.Point3dFromXYZ(target[0]-origin.X-offset.X,target[1]-origin.Y-offset.Y,target[2]-origin.Z-offset.Z));t.Rewrite()
                    targets[int(t.ID64)]=target
            save(journal,{'status':'writing','created_ids':made,'identities':identities,'transport':'native_COM'})
        a.CommandState.StartDefaultCommand();a.ActiveDesignFile.Save();after=snapshot(a);_require_drawing_identity(before,after)
        by={e['id']:e for e in after['elements']}
        assert all(by.get(e['id'])==e for e in before['elements']),'Unrelated element changed'
        for eid,state in existing_tags.items():assert _tag_state(a.ActiveModelReference,eid)==state,('Existing visible label changed',eid)
        assert set(by)-oldids==set(made),'Unexpected native IDs'
        errors=[]
        for eid,target in targets.items():
            t=a.ActiveModelReference.GetElementByID64(eid).AsTagElement;o=t.Origin;d=t.Offset
            errors.append(math.dist([o.X+d.X,o.Y+d.Y,o.Z+d.Z],target))
        assert max(errors,default=0)<1e-7,'Visible label placement differs'
        results=[]
        for c,m,world,receipt in rows:
            eid=identities[c['key']];e=by[eid];tags=[t for t in after['elements']if t.get('base')==eid]
            _require_native_format(e,tags,c);assert _same_native_geometry(e['points'],world,closed=c['kind']not in OPEN_KINDS)
            assert {t['tag']:t['value']for t in tags}=={k:-1.0 if v is None else v for k,v in c['tags'].items()}
            result={'id':eid,'digest':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),'dgn':after['dgn'],'model_name':after['model_name'],
                'native_points':e['points'],'tags':tags,'native_attributes':c['tags'],'command_attributes':c['tags'],
                'status':'native_geometry_and_tags_verified','architectural_acceptance':'not_evaluated','command_contract':'pencil_v1','batch':stage,'transport':'native_COM'}
            save(receipt,result);results.append(result)
        save(journal,{'status':'native_geometry_and_tags_verified','ids':list(identities.values()),'component_count':len(results),'transport':'native_COM','max_visible_tag_position_error_m':max(errors,default=0)})
        return results
    except Exception as exc:
        # Remove only identities created by this invocation, never pre-existing objects.
        for eid in reversed(made):
            try:a.ActiveModelReference.RemoveElement(a.ActiveModelReference.GetElementByID64(eid))
            except Exception:pass
        a.ActiveDesignFile.Save();after=snapshot(a)
        save(journal,{'status':'rolled_back','error':repr(exc),'created_ids':made,'all_original_elements_preserved':all(next((e for e in after['elements']if e['id']==old['id']),None)==old for old in before['elements'])})
        raise
