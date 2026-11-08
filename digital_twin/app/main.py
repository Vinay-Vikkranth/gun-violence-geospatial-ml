import os
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictFloat, StrictInt, ConfigDict
from .model_service import get_service

FRONTEND = Path(__file__).resolve().parents[1] / 'frontend'
app = FastAPI(title='Study Atlas · Gun Violence Digital Twin',version='2.0.0')

class Scenario(BaseModel):
    model_config = ConfigDict(extra='forbid')
    grid_id: str = Field(min_length=1,max_length=40)
    overrides: dict[str, StrictFloat | StrictInt] = Field(default_factory=dict)
    model: str = 'lightgbm'

class SaveScenario(Scenario):
    name: str = Field(min_length=1,max_length=100)

class Sensitivity(Scenario):
    feature: str

class OptimizeRequest(BaseModel):
    model: str = 'lightgbm'
    grid_id: str = Field(min_length=1,max_length=40)
    max_changes: int = Field(default=22, ge=1, le=22)

@app.exception_handler(KeyError)
async def missing(request, exc):
    return JSONResponse(status_code=404,content={'detail':f'Not found: {exc.args[0]}'})

@app.exception_handler(ValueError)
async def invalid(request, exc):
    return JSONResponse(status_code=400,content={'detail':str(exc)})

@app.middleware('http')
async def local_writes(request: Request, call_next):
    if os.environ.get('TWIN_HOSTED') == '1' and request.url.path.startswith('/api/scenarios'):
        return JSONResponse(status_code=403,content={'detail':'Hosted scenarios are saved in your browser.'})
    origin=request.headers.get('origin')
    if request.method in ('POST','DELETE','PUT') and origin and origin != str(request.base_url).rstrip('/'):
        return JSONResponse(status_code=403,content={'detail':'Cross-origin writes are not allowed'})
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Cache-Control']='no-store' if request.url.path.startswith('/api/') else 'no-cache'
    return response

@app.get('/api/health')
def health():
    svc=get_service()
    return dict(ok=True,n_hexes=len(svc.df),model_version=svc.manifest['model_version'])

@app.get('/api/overview')
def overview(): return dict(get_service().overview(), browser_storage=os.environ.get('TWIN_HOSTED') == '1')

@app.get('/api/hexes')
def hexes(city: str | None=None,q: str=Query('',max_length=100),sort: Literal['score','incidents','id']='score',offset: int=Query(0,ge=0),limit: int=Query(60,ge=1,le=200)):
    return get_service().list_hexes(city,q,sort,offset,limit)

@app.get('/api/map/hexes')
def map_hexes(city: str): return get_service().map_hexes(city)

@app.get('/api/hex/{gid}')
def detail(gid: str): return get_service().detail(gid)

@app.post('/api/predict')
def predict(body: Scenario): return get_service().score(body.grid_id,body.overrides,body.model)

@app.post('/api/sensitivity')
def sensitivity(body: Sensitivity): return get_service().sensitivity(body.grid_id,body.feature,body.overrides,body.model)

@app.post('/api/optimize')
def optimize(body: OptimizeRequest): return get_service().optimize(body.grid_id,body.model,body.max_changes)

@app.get('/api/scenarios')
def scenarios(): return get_service().scenarios()

@app.post('/api/scenarios',status_code=201)
def save(body: SaveScenario): return get_service().save(body.name,body.grid_id,body.overrides,body.model)

@app.delete('/api/scenarios/{sid}')
def delete(sid: str):
    get_service().delete(sid)
    return dict(deleted=True)

@app.get('/')
def index(): return FileResponse(FRONTEND/'index.html')

app.mount('/static',StaticFiles(directory=str(FRONTEND)),name='static')
