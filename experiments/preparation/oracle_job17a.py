import argparse, json, time, pathlib
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('base',type=pathlib.Path);parser.add_argument('--raw-dir',type=pathlib.Path,required=True);args=parser.parse_args();base=args.base.resolve();raw_dir=args.raw_dir.resolve();(base/'ducktmp').mkdir(parents=True,exist_ok=True)
import duckdb
start=time.perf_counter(); con=duckdb.connect(':memory:')
con.execute("SET threads=2"); con.execute("SET memory_limit='16GB'")
con.execute("SET temp_directory='"+(base/"ducktmp").as_posix().replace("'", "''")+"'")
cols={
'name':'id name imdb_index imdb_id gender name_pcode_cf name_pcode_nf surname_pcode md5sum',
'cast_info':'id person_id movie_id person_role_id note nr_order role_id',
'title':'id title imdb_index kind_id production_year imdb_id phonetic_code episode_of_id season_nr episode_nr series_years md5sum',
'movie_companies':'id movie_id company_id company_type_id note',
'company_name':'id name country_code imdb_id name_pcode_nf name_pcode_sf md5sum',
'movie_keyword':'id movie_id keyword_id',
'keyword':'id keyword phonetic_code'}
ints=set('id person_id movie_id person_role_id nr_order role_id imdb_id kind_id production_year episode_of_id season_nr episode_nr company_id company_type_id keyword_id'.split())
common="header=false, delim=',', quote='\"', escape='\\', nullstr='', allow_quoted_nulls=false, strict_mode=true"
for t,cs in cols.items():
 p=(raw_dir/(t+'.csv')).as_posix().replace("'","''")
 schema=str({c:('BIGINT' if c in ints else 'VARCHAR') for c in cs.split()})
 con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_csv('{p}',columns={schema},{common})")
sql="""WITH a AS (SELECT ci.movie_id,count(*)::HUGEINT c FROM cast_info ci JOIN name n ON n.id=ci.person_id WHERE n.name LIKE 'B%' GROUP BY ci.movie_id),
b AS (SELECT mc.movie_id,count(*)::HUGEINT c FROM movie_companies mc JOIN company_name cn ON cn.id=mc.company_id WHERE cn.country_code='[us]' GROUP BY mc.movie_id),
c AS (SELECT mk.movie_id,count(*)::HUGEINT c FROM movie_keyword mk JOIN keyword k ON k.id=mk.keyword_id WHERE k.keyword='character-name-in-title' GROUP BY mk.movie_id)
SELECT sum(a.c*b.c*c.c)::HUGEINT,count(*), max(a.c),max(b.c),max(c.c) FROM title t JOIN a ON a.movie_id=t.id JOIN b ON b.movie_id=t.id JOIN c ON c.movie_id=t.id"""
row=con.execute(sql).fetchone()
out={'query':'JOB17a','grouped_count_sql':sql,'count':int(row[0]),'contributing_titles':row[1],'max_branch_counts_on_contributing_titles':list(row[2:]),'seconds':time.perf_counter()-start,'duckdb_version':duckdb.__version__}
original=(base/'job17a.sql').read_text();full_sql='SELECT COUNT(*) '+original[original.index('FROM'):]
out['original_full_predicate_count_sql']=full_sql
out['original_full_predicate_count']=con.execute(full_sql).fetchone()[0]
assert out['original_full_predicate_count']==out['count']
filters={'name':"name LIKE 'B%'",'company_name':"country_code='[us]'",'keyword':"keyword='character-name-in-title'"}
out['local_relation_counts']={t:con.execute(f"SELECT count(*) FROM {t} WHERE {filters.get(t,'true')}").fetchone()[0] for t in cols}
out['feature_audit']={t:con.execute(f'SELECT count(*) FILTER(WHERE {c} IS NULL),min({c}),max({c}) FROM {t}').fetchone() for t,c in [('title','production_year'),('cast_info','nr_order')]}
out['total_seconds']=time.perf_counter()-start
(base/'oracle.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out),flush=True)
