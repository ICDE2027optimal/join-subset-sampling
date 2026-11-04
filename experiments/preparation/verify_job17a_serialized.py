"""Independent DuckDB recomputation of serialized witness moments, never sampler timing."""
import argparse,csv,json,pathlib,time
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('base',type=pathlib.Path);args=parser.parse_args();base=args.base.resolve();import duckdb;out=base/'prepared/JOB17a';start=time.perf_counter();con=duckdb.connect(':memory:')
con.execute('SET threads=2');con.execute("SET memory_limit='16GB'");con.execute("SET temp_directory='"+(base/"ducktmp").as_posix().replace("'", "''")+"'")
for r in range(7):
 key_path=(out/"keys"/f"r{r}.tsv").as_posix().replace("'", "''")
 con.execute(f"CREATE TABLE k{r} AS SELECT * FROM read_csv('{key_path}',header=true,delim='\t',all_varchar=false)")
cases=list(csv.DictReader((out/'cases.tsv').open(),delimiter='\t'));checks=[]
for case in cases:
 for r in range(7):
  path=(out/case['weight_set']/f'r{r}.tsv').as_posix().replace("'","''")
  con.execute(f"CREATE OR REPLACE VIEW r{r} AS SELECT k.*,w.weight::DOUBLE w FROM k{r} k JOIN read_csv('{path}',header=true,delim='\t',columns={{'tuple_id':'BIGINT','weight':'DOUBLE'}}) w USING(tuple_id)")
 sql='''WITH a AS (SELECT edge_1_key x,count(*)::HUGEINT c,sum(r1.w*r0.w) s,sum(pow(r1.w*r0.w,2)) q,max(r1.w*r0.w) m FROM r1 JOIN r0 USING(edge_0_key) GROUP BY edge_1_key),
 b AS (SELECT edge_2_key x,count(*)::HUGEINT c,sum(r3.w*r4.w) s,sum(pow(r3.w*r4.w,2)) q,max(r3.w*r4.w) m FROM r3 JOIN r4 USING(edge_3_key) GROUP BY edge_2_key),
 c AS (SELECT edge_4_key x,count(*)::HUGEINT c,sum(r5.w*r6.w) s,sum(pow(r5.w*r6.w,2)) q,max(r5.w*r6.w) m FROM r5 JOIN r6 USING(edge_5_key) GROUP BY edge_4_key)
 SELECT sum(a.c*b.c*c.c)::HUGEINT,sum(r2.w*a.s*b.s*c.s),sum(r2.w*r2.w*a.q*b.q*c.q),max(r2.w*a.m*b.m*c.m) FROM r2 JOIN a ON r2.edge_1_key=a.x JOIN b ON r2.edge_2_key=b.x JOIN c ON r2.edge_4_key=c.x'''
 row=con.execute(sql).fetchone();assert row[0]==int(case['join_size'])
 errors={}
 for field,val in zip(['actual_mu','sum_p2','max_p'],row[1:]):
  expected=float(case[field]);err=abs(val-expected)/max(abs(expected),1e-300);assert err<1e-10,(field,val,expected,err);errors[field]=err
 checks.append({'mode':case['mode'],'requested_targets':case['requested_targets'],'count':int(row[0]),'mu':row[1],'sum_p2':row[2],'max_p':row[3],'relative_errors':errors});print(checks[-1],flush=True)
(base/'serialized_validation.json').write_text(json.dumps({'independent_engine':'DuckDB '+duckdb.__version__,'all_serialized_moments_match':True,'checks':checks,'seconds':time.perf_counter()-start},indent=2)+'\n')
