"""Stage-sized pencil transactions: explicit geometry only, one native macro.

No source interpretation, snapping, classification or architectural review.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from shapely.geometry import Polygon, LineString
from .command_contract import prepare_command, validate_transport, canonical_points
from .native import app, snapshot, RasterMap, save
from .native_schema import LEVELS, TAGS, OPEN_KINDS
from .writer import quote, value_vba, native_attributes


def _require_drawing_identity(expected, actual):
    if (Path(actual['dgn']).resolve() != Path(expected['dgn']).resolve()
            or actual['unit'] != 'm' or actual['rasters'] != expected['rasters']
            or ('model_name' in expected and actual.get('model_name') != expected['model_name'])):
        raise RuntimeError('Active drawing identity, model or registration changed')


def _require_native_format(element, attached, command):
    """Check the fixed native representation, without interpreting geometry."""
    kind = command['kind']
    level, number, _ = LEVELS[kind]
    types = {3, 4} if kind in OPEN_KINDS else {6}
    if (element.get('level') != level or element.get('level_number') != number
            or element.get('type') not in types):
        raise RuntimeError('Native component type/level differs: ' + command['key'])
    definitions = {name: (taglevel, tagnum) for name, _, taglevel, tagnum
                   in TAGS.get(kind, ('', []))[1]}
    if len(attached) != len(definitions) or {t.get('tag') for t in attached} != set(definitions):
        raise RuntimeError('Native linked Tag set differs: ' + command['key'])
    for tag in attached:
        expected = definitions.get(tag.get('tag'))
        if (expected is None or tag.get('type') != 37
                or (tag.get('level'), tag.get('level_number')) != expected):
            raise RuntimeError('Native linked Tag type/level differs: ' + command['key'])


def _same_native_geometry(points, world, *, closed):
    """Exact vertex geometry, allowing ring start/direction and native rounding."""
    if any(len(p) > 2 and abs(p[2]) > 1e-8 for p in points):
        return False
    actual = canonical_points([p[:2] for p in points], closed=closed)
    if len(actual) != len(world):
        return False
    def matches(candidate):
        return all(math.dist(p, q) <= 1e-8 for p, q in zip(candidate, world))
    if not closed:
        return matches(actual) or matches(actual[::-1])
    return any(matches(ordered[i:] + ordered[:i])
               for ordered in (actual, actual[::-1]) for i in range(len(actual)))


def write_batch(commands, initial, directory, stage):
    # Legacy callers retain the small-macro path; large native batches must
    # never reach VBA's per-procedure bytecode limit. FloorCanvas uses COM for
    # every batch, including small ones.
    if len(commands)>8 or sum(len(c.get('vertices_raw',[]))for c in commands)>160 or len(json.dumps(commands).encode('utf8'))>12000:
        from .native_batch_writer import write_batch_com
        return write_batch_com(commands,initial,directory,stage)
    import comtypes.client
    if not stage.replace('_', '').isalnum():
        raise ValueError('Invalid stage key')
    application = app()
    live = snapshot(application)
    _require_drawing_identity(initial, live)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prepared = [prepare_command(c, initial) for c in commands]
    if len({c['key'] for c in prepared}) != len(prepared):
        raise ValueError('Duplicate component key')
    for c in prepared:
        validate_transport(c, initial)
        if Path(c.get('source_dgn', live['dgn'])).resolve() != Path(live['dgn']).resolve():
            raise RuntimeError('Command belongs to another drawing')
    checked_sources={}
    for c in prepared:
        raster=live['rasters'][c.get('source_raster_index',1)-1]
        if c.get('source_sha256'):
            path=Path(raster['tif'])
            if path not in checked_sources:
                with path.open('rb') as stream:
                    checked_sources[path]=hashlib.file_digest(stream,'sha256').hexdigest()
            if c['source_sha256']!=checked_sources[path]:
                raise RuntimeError('Source pixels changed')
    rows = []
    pending = []
    for c in prepared:
        mapping=RasterMap(live['rasters'][c.get('source_raster_index',1)-1])
        digest = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
        receipt = directory / f"{c['key']}_native.json"
        attrs = native_attributes(c)
        world = [mapping.xy(*p) for p in c['vertices_raw']]
        row = dict(command=c, digest=digest, receipt=receipt, attrs=attrs, world=world,mapping=mapping)
        if receipt.exists():
            prior = json.loads(receipt.read_text(encoding='utf8'))
            e = next((e for e in live['elements'] if e['id'] == prior['id']), None)
            tags = [e for e in live['elements'] if e.get('base') == prior['id']]
            if (prior['digest'] != digest or prior['dgn'] != live['dgn'] or not e
                    or e.get('points') != prior['native_points'] or tags != prior['tags']
                    or ('model_name' in prior and prior['model_name'] != live.get('model_name'))):
                raise RuntimeError('Preserve edited/missing previous component: ' + c['key'])
            _require_native_format(e, tags, c)
            row['prior'] = prior
        else:
            level = LEVELS[c['kind']][0]
            closed = c['kind'] not in OPEN_KINDS
            duplicate = next((e for e in live['elements'] if e.get('level') == level
                              and e.get('points') and _same_native_geometry(e['points'], world, closed=closed)), None)
            if duplicate is not None:
                raise RuntimeError('Duplicate native geometry already exists; preserve existing element '
                                   + str(duplicate['id']) + ': ' + c['key'])
            if any(LEVELS[r['command']['kind']][0] == level
                   and _same_native_geometry(r['world'], world, closed=closed) for r in pending):
                raise RuntimeError('Duplicate geometry in batch: ' + c['key'])
            pending.append(row)
        rows.append(row)
    if not pending:
        return [r['prior'] for r in rows]
    journal = directory / f'{stage}_transaction.json'
    if journal.exists() and json.loads(journal.read_text(encoding='utf8'))['status'] == 'prepared':
        raise RuntimeError('Unfinished stage journal; inspect instead of duplicating')
    application.ActiveDesignFile.Save()
    backup = directory / 'before_first_batch.dgn'
    if not backup.exists():
        shutil.copy2(live['dgn'], backup)
    save(journal, dict(status='prepared', dgn=live['dgn'], model_name=live.get('model_name'),
                      commands=[r['command'] for r in pending], before_ids=[e['id'] for e in live['elements']]))
    requirements = {}
    for row in pending:
        c = row['command']; kind = c['kind']
        level, number, _ = LEVELS[kind]
        requirements[level] = number
        for _, _, name, number in TAGS.get(kind, ('', []))[1]:
            requirements[name] = number
        if kind in TAGS:
            name, defs = TAGS[kind]
            try: ts = application.ActiveDesignFile.TagSets.Item(name)
            except Exception: ts = application.ActiveDesignFile.TagSets.Add(name)
            for tag, tagtype, _, _ in defs:
                try: ts.TagDefinitions.Item(tag)
                except Exception: ts.TagDefinitions.Add(tag, tagtype)
    known = {lv.Name: lv for lv in application.ActiveDesignFile.Levels}
    occupied = {lv.Number: lv.Name for lv in application.ActiveDesignFile.Levels}
    for name, number in requirements.items():
        if name not in known:
            if number in occupied:
                raise RuntimeError('Level number occupied: ' + str(number))
            lv = application.ActiveDesignFile.AddNewLevel(name); lv.Number = number
        elif known[name].Number != number:
            raise RuntimeError('Unexpected native level number: ' + name)
    application.ActiveDesignFile.RewriteLevels()
    done, error = directory / f'{stage}.ids', directory / f'{stage}.error'
    for path in (done, error):
        if path.exists(): path.unlink()
    code = ['Option Explicit', 'Public Sub WriteStage()', 'On Error GoTo Failed',
        'Dim p() As Point3d, e As Element, t As TagElement, ts As TagSet, sty As TextStyle',
        'Dim f As Integer, j As Long, made As New Collection, message As String',
        f'If LCase$(ActiveDesignFile.FullName) <> LCase$({quote(live["dgn"])}) Then Err.Raise 5']
    if 'model_name' in live:
        code.append(f'If ActiveModelReference.Name <> {quote(live["model_name"])} Then Err.Raise 5')
    code += ['Set sty = ActiveSettings.TextStyle', 'f = FreeFile', f'Open {quote(done)} For Output As #f']
    for row in pending:
        c, world, attrs = row['command'], row['world'], row['attrs']
        kind = c['kind']; level, _, color = LEVELS[kind]
        code.append(f'ReDim p(0 To {len(world)-1})')
        code.extend(f'p({i}) = Point3dFromXYZ({x:.12f},{y:.12f},0)' for i, (x, y) in enumerate(world))
        create = 'CreateLineElement1(Nothing,p)' if kind in OPEN_KINDS else 'CreateShapeElement1(Nothing,p,msdFillModeNotFilled)'
        code += ['Set e = ' + create, f'Set e.Level = ActiveDesignFile.Levels({quote(level)})',
                 f'e.Color = {color}: e.LineWeight = 1']
        if kind == 'window':
            code.append('Set e.LineStyle = ActiveDesignFile.LineStyles("2")')
        code += ['ActiveModelReference.AddElement e', 'made.Add e']
        if kind in TAGS:
            tagset, definitions = TAGS[kind]
            geom = LineString(c['vertices_raw']) if kind in OPEN_KINDS else Polygon(c['vertices_raw'])
            position = c.get('label_raw', list(geom.representative_point().coords)[0])
            x, y = row['mapping'].xy(*position)
            code += [f'Set ts = ActiveDesignFile.TagSets({quote(tagset)})',
                     'sty.Height = .20: sty.Width = .20', 'Set sty.Font = ActiveDesignFile.Fonts("Arial")',
                     'Set ActiveSettings.TextStyle = sty']
            for i, (tag, _, taglevel, _) in enumerate(definitions):
                code += [f'Set t = e.AddTag(ts.TagDefinitions({quote(tag)}))', 'made.Add t',
                    f't.Value = {value_vba(attrs[tag])}', f'Set t.Level = ActiveDesignFile.Levels({quote(taglevel)})',
                    f't.Move Point3dSubtract(Point3dFromXYZ({x:.12f},{y-i*.23:.12f},0),t.Origin)',
                    f't.Color = {color}',
                    f't.IsHidden = {"True" if attrs[tag] is None or attrs[tag] in ("", "-1") else "False"}', 't.Rewrite']
        code.append(f'Print #f, {quote(c["key"])} & vbTab & DLongToString(e.ID)')
    code += ['Close #f: f = 0', 'ActiveDesignFile.Save', 'ActiveDesignFile.Views(1).Redraw', 'Exit Sub',
        'Failed:', 'message = Err.Number & " " & Err.Description', 'On Error Resume Next',
        'If f <> 0 Then Close #f', 'For j = made.Count To 1 Step -1',
        'ActiveModelReference.RemoveElement made(j)', 'Next j', 'ActiveDesignFile.Save',
        'f = FreeFile', f'Open {quote(error)} For Output As #f', 'Print #f, message', 'Close #f', 'End Sub']
    host = __import__('config').PROJECTS / 'sinir_otomasyonu/SinirPilot.mvba'
    application.CadInputQueue.SendKeyin('vba load ' + str(host))
    project = None
    for _ in range(30):
        try:
            vbe = comtypes.client.GetBestInterface(application.VBE)
            project = next((vbe.VBProjects.Item(i) for i in range(1, vbe.VBProjects.Count+1)
                            if vbe.VBProjects.Item(i).Name == 'SinirPilot'), None)
        except Exception: pass
        if project is not None: break
        time.sleep(.1)
    if project is None: raise RuntimeError('Native macro host unavailable')
    component = next((project.VBComponents.Item(i) for i in range(1, project.VBComponents.Count+1)
                      if project.VBComponents.Item(i).Name == 'ArchitectBatch'), None)
    if component is None: component = project.VBComponents.Add(1); component.Name = 'ArchitectBatch'
    module = component.CodeModule
    if module.CountOfLines: module.DeleteLines(1, module.CountOfLines)
    source = '\r\n'.join(code)
    (directory / f'{stage}.bas').write_text(source, encoding='utf8')
    module.AddFromString(source)
    application.CadInputQueue.SendKeyin('vba run [SinirPilot]ArchitectBatch.WriteStage')
    expected_count = len(pending)
    for _ in range(300):
        if error.exists():
            save(journal, dict(status='rolled_back', error=error.read_text(errors='replace')))
            raise RuntimeError(error.read_text(errors='replace'))
        if done.exists() and len(done.read_text().strip().splitlines()) == expected_count:
            # Final save is synchronous in the queued macro; document API waits for it.
            break
        time.sleep(.1)
    else: raise RuntimeError('Batch bridge did not complete; retain prepared journal')
    after = snapshot(application)
    _require_drawing_identity(live, after)
    after_by_id = {e['id']: e for e in after['elements']}
    if any(after_by_id.get(e['id']) != e for e in live['elements']):
        raise RuntimeError('Existing native elements changed during batch; inspect prepared journal')
    identities = dict(line.split('\t') for line in done.read_text().strip().splitlines())
    results = []
    for row in rows:
        if 'prior' in row:
            results.append(row['prior']); continue
        c = row['command']; eid = int(identities[c['key']])
        element = next(e for e in after['elements'] if e['id'] == eid)
        attached = [e for e in after['elements'] if e.get('base') == eid]
        _require_native_format(element, attached, c)
        actual = canonical_points([p[:2] for p in element['points']], closed=c['kind'] not in OPEN_KINDS)
        if len(actual) != len(row['world']) or any(math.dist(p, q) > 1e-8 for p, q in zip(actual, row['world'])):
            raise RuntimeError('Batch native geometry differs: ' + c['key'])
        if {t['tag']: t['value'] for t in attached} != {k: -1.0 if v is None else v for k, v in row['attrs'].items()}:
            raise RuntimeError('Batch native attributes differ: ' + c['key'])
        result = dict(id=eid, digest=row['digest'], dgn=after['dgn'], native_points=element['points'],
            model_name=after.get('model_name'),
            tags=attached, native_attributes=row['attrs'], command_attributes=c['tags'],
            status='native_geometry_and_tags_verified', architectural_acceptance='not_evaluated',
            command_contract='pencil_v1', batch=stage)
        save(row['receipt'], result); results.append(result)
    save(journal, dict(status='native_geometry_and_tags_verified', ids=[r['id'] for r in results],
                      component_count=len(results), architecture_review='not_performed'))
    return results
