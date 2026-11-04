#!/usr/bin/env python3
"""JOB17a fixed full-witness inputs. Uses the original robust COPY parser."""
import argparse, array, hashlib, itertools, json, os, pathlib, shutil, sys, time
import numpy as np

p=argparse.ArgumentParser();p.add_argument('base',type=pathlib.Path);p.add_argument('--stage',choices=['parse','prepare'],required=True);p.add_argument('--raw-dir',type=pathlib.Path,required=True);a=p.parse_args()
base=a.base.resolve();base.mkdir(parents=True,exist_ok=True);raw_dir=a.raw_dir.resolve()
import prepare_job as orig
tables=['name','cast_info','title','movie_companies','company_name','movie_keyword','keyword']
aliases=['n','ci','t','mc','cn','mk','k']; widths=[9,7,12,5,7,3,3]
edge_specs=[(0,1,'id','person_id',0,1),(1,2,'movie_id','id',2,0),(2,3,'id','movie_id',0,1),(3,4,'company_id','id',2,0),(2,5,'id','movie_id',0,1),(5,6,'keyword_id','id',2,0)]
query={'id':'JOB17a','verification_root':2,'anchor_relation':2,'relations':[{'id':i,'alias':aliases[i],'table':t} for i,t in enumerate(tables)],'edges':[{'id':i,'left':l,'right':r,'left_columns':[lc],'right_columns':[rc]} for i,(l,r,lc,rc,lp,rp) in enumerate(edge_specs)]}
neighbors=orig.tree_neighbors(query); eps=1/64
def dump(name,x):
 (base/name).write_text(json.dumps(x,indent=2)+'\n')
def progress(s):print(s,flush=True)
if a.stage=='parse':
 (base/'cache').mkdir(exist_ok=True); records=[]
 for rid,t in enumerate(tables):
  start=time.perf_counter();path=raw_dir/(t+'.csv')
  ids=array.array('q'); keys={eid:array.array('q') for _,eid in neighbors[rid]}; feature=array.array('i'); raw_count=0; nulls=0; empties=0; fmin=None;fmax=None
  last_id=0; order=True
  for line,row in orig.csv_rows(path,widths[rid]):
   raw_count+=1; rowid=int(row[0]); order &= rowid>last_id;last_id=rowid
   if rid==0 and not (row[1] is not None and row[1].startswith('B')):continue
   if rid==4 and row[2]!='[us]':continue
   if rid==6 and row[1]!='character-name-in-title':continue
   ids.append(rowid)
   for _,eid in neighbors[rid]:
    l,r,lc,rc,lp,rp=edge_specs[eid]; value=row[lp if l==rid else rp]
    if value is None:raise ValueError('Unexpected NULL in declared NOT NULL join key')
    keys[eid].append(int(value))
   if rid in (1,2):
    value=row[5 if rid==1 else 4];nulls+=value is None;empties+=value==''
    val=-(1<<31) if value is None else int(value);feature.append(val)
    if value is not None:fmin=val if fmin is None else min(fmin,val);fmax=val if fmax is None else max(fmax,val)
  if len(np.unique(np.asarray(ids,dtype=np.int64)))!=len(ids):raise AssertionError('Retained IDs must be unique')
  np.save(base/'cache'/f'r{rid}_ids.npy',np.asarray(ids,dtype=np.int64))
  for eid,v in keys.items():np.save(base/'cache'/f'r{rid}_raw{eid}.npy',np.asarray(v,dtype=np.int64))
  if rid in (1,2):np.save(base/'cache'/f'r{rid}_feature.npy',np.asarray(feature,dtype=np.int32))
  rec={'relation_id':rid,'table':t,'source_rows':raw_count,'retained_rows':len(ids),'retained_ids_unique':True,'source_ids_increasing':order,'feature_nulls':nulls,'feature_quoted_empty':empties,'feature_min':fmin,'feature_max':fmax,'source_bytes':path.stat().st_size,'seconds':time.perf_counter()-start,'sha256':orig.sha256_file(path)}
  records.append(rec);dump('inventory.json',records);progress(rec)
 sys.exit(0)

start=time.perf_counter();out=base/'prepared/JOB17a';out.mkdir(parents=True,exist_ok=True)
inventory=json.loads((base/'inventory.json').read_text());oracle=json.loads((base/'oracle.json').read_text())
for row in inventory:
 assert row['retained_rows']==oracle['local_relation_counts'][row['table']]
 if row['relation_id'] in (1,2):assert [row['feature_nulls'],row['feature_min'],row['feature_max']]==oracle['feature_audit'][row['table']]
assert orig.parse_job_csv_line(',"",x',context='tiny-null')==[None,'','x']
stores=[]
for rid in range(7):
 s={'tuple_ids':np.load(base/'cache'/f'r{rid}_ids.npy',mmap_mode='r'),'raw_keys':{eid:np.load(base/'cache'/f'r{rid}_raw{eid}.npy',mmap_mode='r') for _,eid in neighbors[rid]},'edge_keys':{}}
 if rid in (1,2):s['feature']=np.load(base/'cache'/f'r{rid}_feature.npy',mmap_mode='r')
 stores.append(s)
card=orig.intern_edge_keys(stores,query);progress('interned edge keys')
timings={'intern_seconds':time.perf_counter()-start}
def exact(st):
 # Dimension IDs are unique. Each leaf message is0/1, so each branch total is at most its source N.
 vals=[]
 for leaf,fact,le,te in [(0,1,0,1),(4,3,3,2),(6,5,5,4)]:
  support=np.zeros(card[le]+1,dtype=np.uint64);np.add.at(support,st[leaf]['edge_keys'][le],np.uint64(1))
  grouped=np.zeros(card[te]+1,dtype=np.uint64);np.add.at(grouped,st[fact]['edge_keys'][te],support[st[fact]['edge_keys'][le]])
  vals.append(grouped[st[2]['edge_keys'][te]])
 return sum(int(x)*int(y)*int(z) for x,y,z in zip(*vals)),vals
J,branch=exact(stores);oracle=json.loads((base/'oracle.json').read_text());assert J==oracle['count'] and J>0
progress({'integer_dp_count':J,'oracle_match':True})
weights=[np.ones(len(s['tuple_ids']),dtype=np.float64) for s in stores]
years=stores[2]['feature'];yw=np.full(len(years),eps);valid=years!=-(1<<31);yw[valid]=eps+(1-eps)*(years[valid].astype(float)-1880)/(2019-1880);assert np.all((yw>=eps)&(yw<=1))
order=stores[1]['feature'];cw=np.full(len(order),eps);valid=order>0;cw[valid]=eps+(1-eps)/order[valid];assert np.all((cw>=eps)&(cw<=1))

# Tiny induced actual-source witness with two occurrences per fact branch.
ti=int(np.argmax(branch[0]*branch[1]*branch[2]));sel={2:[ti]}
for leaf,fact,le,te in [(0,1,0,1),(4,3,3,2),(6,5,5,4)]:
 candidates=np.flatnonzero(stores[fact]['edge_keys'][te]==stores[2]['edge_keys'][te][ti]);leafmap={int(k):i for i,k in enumerate(stores[leaf]['edge_keys'][le])}; chosen=[int(i) for i in candidates if int(stores[fact]['edge_keys'][le][i]) in leafmap][:2];sel[fact]=chosen;sel[leaf]=sorted({leafmap[int(stores[fact]['edge_keys'][le][i])] for i in chosen})
tiny=[{'tuple_ids':s['tuple_ids'][sel[r]],'edge_keys':{e:v[sel[r]] for e,v in s['edge_keys'].items()}} for r,s in enumerate(stores)]
tw=[w[sel[r]].copy() for r,w in enumerate(weights)];tw[1]=cw[sel[1]];tw[2]=yw[sel[2]]
witnesses=[]
for ids in itertools.product(*(range(len(s['tuple_ids'])) for s in tiny)):
 if all(tiny[e['left']]['edge_keys'][e['id']][ids[e['left']]]==tiny[e['right']]['edge_keys'][e['id']][ids[e['right']]] for e in query['edges']):witnesses.append(ids)
tinyJ,_=exact(tiny);assert tinyJ==len(witnesses)
tm=orig.all_root_metrics_compact(tiny,query,card,tw,tinyJ)
ps=[np.prod([np.longdouble(tw[r][ids[r]]) for r in range(7)]) for ids in witnesses]
for r in tm['roots']:
 assert np.isclose(r['mu'],sum(ps),rtol=1e-13);assert np.isclose(r['sum_p2'],sum(x*x for x in ps),rtol=1e-13);assert np.isclose(r['max_p'],max(ps),rtol=1e-13)
 env=sum(sum(1 for ids in witnesses if ids[r['root']]==i)*max([ps[j] for j,ids in enumerate(witnesses) if ids[r['root']]==i],default=0) for i in range(len(tiny[r['root']]['tuple_ids'])))
 assert np.isclose(env,r['lambda'],rtol=1e-13)
dump('tiny_validation.json',{'count':tinyJ,'all_root_moments_envelopes_match_enumeration':True,'tuple_ids':[[int(v) for v in s['tuple_ids']] for s in tiny]});del tm,tiny,branch

st=time.perf_counter();structure=['stats-undirected-v1','query\tJOB17a','m\t7','verification_root\t2'];degree=[]
for rid,s in enumerate(stores):
 structure.append('\t'.join(['relation',str(rid),aliases[rid],tables[rid],str(len(s['tuple_ids'])),f'keys/r{rid}.tsv',f'features/r{rid}.tsv']))
 eids=sorted(s['edge_keys']);orig.write_numeric_tsv(out/'keys'/f'r{rid}.tsv',['tuple_id']+[f'edge_{e}_key' for e in eids],([int(s['tuple_ids'][i])]+[int(s['edge_keys'][e][i]) for e in eids] for i in range(len(s['tuple_ids']))))
 if rid in (1,2):orig.write_numeric_tsv(out/'features'/f'r{rid}.tsv',['tuple_id','nr_order' if rid==1 else 'production_year'],((int(t),'\\N' if f==-(1<<31) else int(f)) for t,f in zip(s['tuple_ids'],s['feature'])))
 else:orig.write_numeric_tsv(out/'features'/f'r{rid}.tsv',['tuple_id'],((int(t),) for t in s['tuple_ids']))
 for e in eids:
  freq=np.bincount(s['edge_keys'][e],minlength=card[e]+1);nz=freq[freq>0];degree.append({'rid':rid,'edge':e,'nonempty_keys':len(nz),'quantiles':dict(zip(['min','p50','p90','p99','max'],map(float,np.quantile(nz,[0,.5,.9,.99,1]))))})
 progress(f'wrote relation {rid}')
for e in query['edges']:structure.append('\t'.join(['edge',str(e['id']),str(e['left']),str(e['right']),str(card[e['id']]),','.join(e['left_columns']),','.join(e['right_columns'])]))
(out/'structure.tsv').write_text('\n'.join(structure)+'\n');timings['write_shared_seconds']=time.perf_counter()-st
cases=[];rootrows=[];modes={}; canonical={};weights_keep=[]
def weightfiles(relative,ws):
 for rid,w in enumerate(ws):
  path=out/relative/f'r{rid}.tsv';path.parent.mkdir(parents=True,exist_ok=True)
  # Sharing identical weight files preserves the standard loader interface.
  identity=id(w)
  if identity in canonical:
   try:os.link(canonical[identity],path)
   except OSError:shutil.copyfile(canonical[identity],path)
  else:
   orig.write_numeric_tsv(path,['tuple_id','weight'],((int(t),format(float(v),'.17g')) for t,v in zip(stores[rid]['tuple_ids'],w)));canonical[identity]=path
for mode,targets in [('product-real',[1,100,10000]),('uniform',[100]),('single-anchor',[100])]:
 ws=list(weights)
 if mode!='uniform':ws[2]=yw
 if mode=='product-real':ws[1]=cw
 st=time.perf_counter();metrics=orig.all_root_metrics_compact(stores,query,card,ws,J);timings[mode+'_metrics_seconds']=time.perf_counter()-st
 chosen=metrics['roots'][metrics['chosen_root']];mu0=chosen['mu'];groups={}
 for target in targets:groups.setdefault(min(1.,target/float(mu0)),[]).append(target)
 modes[mode]={'chosen_root':metrics['chosen_root'],'mu0':orig.format_ld(mu0),'roots':[{k:(orig.format_ld(v) if isinstance(v,np.floating) else v) for k,v in r.items()} for r in metrics['roots']]}
 for r in metrics['roots']:rootrows.append([mode,r['root'],orig.format_ld(r['lambda']),orig.format_ld(r['kappa']),orig.format_ld(r['mu']),int(r['root']==metrics['raw_min_root']),int(r['root'] in metrics['near_tie_eligible_roots']),int(r['root']==metrics['chosen_root'])])
 st=time.perf_counter()
 for idx,(scale,targets2) in enumerate(sorted(groups.items())):
  relative=pathlib.Path('weights')/mode/f'scale-{idx:03d}';scaled=list(ws);scaled[2]=ws[2]*scale
  # Retain arrays until all files are written, preventing object-id reuse.
  weights_keep.append(scaled[2])
  weightfiles(relative,scaled)
  cases.append([mode,','.join(map(str,targets2)),relative.as_posix(),J,format(scale,'.17g'),orig.format_ld(mu0),orig.format_ld(mu0*scale),orig.format_ld(chosen['sum_p2']*np.longdouble(scale)**2),orig.format_ld(chosen['max_p']*scale),metrics['chosen_root']])
 timings[mode+'_write_weights_seconds']=time.perf_counter()-st;progress({'mode':mode,'mu0':str(mu0),'chosen_root':metrics['chosen_root']})
orig.write_numeric_tsv(out/'cases.tsv',['mode','requested_targets','weight_set','join_size','a','mu0','actual_mu','sum_p2','max_p','chosen_root'],cases)
orig.write_numeric_tsv(out/'root_scores.tsv',['mode','root','lambda','kappa','mu','raw_min','near_tie_eligible','guarded_chosen'],rootrows)
timings['total_prepare_seconds']=time.perf_counter()-start
dump('validation.json',{'query':query,'N':sum(len(s['tuple_ids']) for s in stores),'J':J,'integer_dp_grouped_sql_match':True,'degree_quantiles':degree,'modes':modes,'timings':timings,'query_sql_sha256':orig.sha256_file(base/'job17a.sql'),'feature_protocol':'Title recency and inverse positive cast nr_order, neutral1 elsewhere, selected before counts and timing','settings':len(cases)})
dump('prepared_manifest.json',{'query':'JOB17a','prepared_directory':str(out),'format':'stats-undirected-v1','prepared_files':orig.hash_files(out),'code_sha256':orig.sha256_file(pathlib.Path(__file__)),'original_parser_sha256':orig.sha256_file(pathlib.Path(orig.__file__))})
progress({'complete':True,'N':sum(len(s['tuple_ids']) for s in stores),'J':J,'settings':len(cases)})
