"""The pencil boundary: explicit geometry and native values, no architecture.

No source detection, snapping, classification, neighbour search or policy lookup
is permitted here. Validation describes what the native format can represent;
it does not certify that a drawing is architecturally correct.
"""
from copy import deepcopy
import math
import re

from shapely.geometry import LineString, Polygon
from .native_schema import LEVELS, OPEN_KINDS, TAGS


def canonical_points(points, *, closed):
    """Drop only an exact repeated closing vertex of an explicitly closed type."""
    result=[list(p) for p in points]
    if closed and len(result)>1 and result[0]==result[-1]:
        result.pop()
    return result


def _point(value):
    if not isinstance(value,(list,tuple)) or len(value)!=2:
        raise ValueError('Each coordinate must be an explicit 2D point')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in value):
        raise ValueError('Coordinates must be finite numbers')


def explicit_attributes(command):
    """Copy final native attributes. Never consult the drawing or firm rules."""
    kind=command.get('kind')
    if kind not in LEVELS:raise ValueError('Unknown native component kind')
    if kind=='window' and command.get('sill_reference','from_floor') not in {'from_floor','native'}:
        raise ValueError('Writer requires a resolved native sill; convert the explicit ceiling offset before writing')
    attrs=command.get('tags',{})
    definitions=TAGS.get(kind,('',[]))[1]
    if not isinstance(attrs,dict) or set(attrs)!={r[0] for r in definitions}:
        raise ValueError('Incomplete or unexpected native attributes')
    for name,tagtype,_,_ in definitions:
        value=attrs[name]
        if tagtype==1:
            if not isinstance(value,str) or any(ord(c)<32 for c in value):
                raise ValueError(name+' must be native text without control characters')
        elif tagtype==3:
            if (isinstance(value,bool) or not isinstance(value,(int,float))
                or not math.isfinite(value) or value!=int(value)
                or not -(2**31)<=value<2**31):
                raise ValueError(name+' must be a representable integer in centimetres')
        elif value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):
            raise ValueError(name+' must be a finite number or null')
    return deepcopy(attrs)


def prepare_command(command, initial=None):
    """Return a validated copy; the only normalization is ring serialization."""
    if not isinstance(command,dict):raise ValueError('Command must be an object')
    key=command.get('key')
    if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_]{1,80}',key):
        raise ValueError('Invalid command identity')
    attrs=explicit_attributes(command)
    points=command.get('vertices_raw')
    if not isinstance(points,(list,tuple)):raise ValueError('Explicit vertices_raw required')
    for p in points:_point(p)
    opened=command['kind'] in OPEN_KINDS
    points=canonical_points(points,closed=not opened)
    if len(points)<(2 if opened else 3):raise ValueError('Too few native vertices')
    geometry=LineString(points) if opened else Polygon(points)
    if not geometry.is_valid or (geometry.length<=0 if opened else geometry.area<=0):
        raise ValueError('Geometry cannot be represented as the requested native primitive')
    label=command.get('label_raw')
    if label is not None:_point(label)
    if initial is not None:
        raster_index=command.get('source_raster_index',1)
        if isinstance(raster_index,bool) or not isinstance(raster_index,int) or not 1<=raster_index<=len(initial['rasters']):
            raise ValueError('Invalid explicit source raster index')
        w,h=initial['rasters'][raster_index-1]['pixels']
        if any(not (0<=x<=w and 0<=y<=h) for x,y in points+([label] if label is not None else [])):
            raise ValueError('Coordinate outside registered source image')
    result=deepcopy(command)
    result.update(vertices_raw=points,tags=attrs)
    return result


def validate_transport(command, initial=None):
    try:
        prepared=prepare_command(command,initial)
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        return {'eligible':False,'errors':[str(exc)],'scope':'transport_only',
                'architectural_acceptance':'not_evaluated'}
    return {'eligible':True,'errors':[],'scope':'transport_only',
            'architectural_acceptance':'not_evaluated',
            'normalizations':(['exact_closing_vertex_representation']
                if prepared['vertices_raw']!=command['vertices_raw'] else [])}
