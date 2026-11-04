"""Exact bag-count via grouped SQL, independently parsing raw source CSV."""
import argparse,json,time
from pathlib import Path
a=argparse.ArgumentParser()
for flag in ('raw-dir','source-manifest','query-manifest','temp-dir'):a.add_argument('--'+flag,required=True,type=Path)
args=a.parse_args();import duckdb;start=time.perf_counter()
c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='8GiB'")
args.temp_dir.mkdir(parents=True,exist_ok=True)
c.execute("SET temp_directory='"+str(args.temp_dir).replace("'","''")+"'")
for f in json.loads(args.source_manifest.read_text())['files']:
    if f['role']!='table':continue
    cols=', '.join("'%s':'%s'"%(n,{'int':'BIGINT','timestamp':'TIMESTAMP'}[t]) for n,t in f['columns'].items())
    path=str(args.raw_dir/f['name']).replace("'","''")
    c.execute(f'''CREATE TABLE "{f['table']}" AS SELECT * FROM read_csv('{path}',header=true,columns={{{cols}}},nullstr=['','\\N'])''')
parse=time.perf_counter()-start;start=time.perf_counter()
sql='''WITH ph AS (SELECT PostId, count(*)::HUGEINT n FROM postHistory GROUP BY PostId),
pl AS (SELECT PostId,count(*)::HUGEINT n FROM postLinks GROUP BY PostId),
b AS (SELECT UserId,count(*)::HUGEINT n FROM badges GROUP BY UserId),
cp AS (SELECT c.UserId,sum(pl.n)::HUGEINT n FROM comments c JOIN pl ON c.PostId=pl.PostId GROUP BY c.UserId),
vp AS (SELECT v.UserId,sum(ph.n)::HUGEINT n FROM votes v JOIN ph ON v.PostId=ph.PostId GROUP BY v.UserId)
SELECT coalesce(sum(vp.n*cp.n*b.n),0)::HUGEINT FROM vp JOIN cp ON vp.UserId=cp.UserId JOIN b ON b.UserId=vp.UserId'''
count=int(c.execute(sql).fetchone()[0]);elapsed=time.perf_counter()-start
(args.temp_dir.parent/'oracle_details.json').write_text(json.dumps({'sql':sql,'raw_csv_parse_seconds':parse,'exact_grouped_count_seconds':elapsed,'count':str(count),'threads':2,'memory_limit':'8GiB'},indent=2)+'\n')
print(json.dumps({'duckdb_version':duckdb.__version__,'counts':{'NEWbranch5':str(count)}}))
