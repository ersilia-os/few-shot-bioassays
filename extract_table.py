import sys
import psycopg2
import pandas as pd
import pandas.io.sql as sqlio

def chembl_activity_target(db_user, db_password, db_name='chembl_33', 
						   	db_host='localhost', db_port=5432):
	conn = psycopg2.connect(user=db_user
							,password=db_password
							,host=db_host
							,port=db_port
							,database=db_name)
	# conn.set_session(readonly=True, autocommit=True)
	cursor = conn.cursor()
	query = """
		CREATE TEMP TABLE temp_bioassay_table AS
		SELECT 
			a.assay_id,
			cnd_s.canonical_smiles, 
			act.pchembl_value
		FROM target_dictionary td
		INNER JOIN assays a ON td.tid = a.tid
		INNER JOIN activities act ON a.assay_id = act.assay_id
		INNER JOIN molecule_dictionary md ON act.molregno = md.molregno
		INNER JOIN compound_structures cnd_s ON md.molregno = cnd_s.molregno
		WHERE act.pchembl_value > 0
		"""
	cursor.execute(query)

	cursor.execute("SELECT count(*) FROM temp_bioassay_table")
	print(f'{cursor.fetchone()[0]} rows extracted from Chembl')
	cursor.execute("SELECT COUNT(DISTINCT assay_id) FROM temp_bioassay_table")
	print(f'{cursor.fetchone()[0]} unique assays extracted from Chembl')


	# sql = "SELECT * FROM temp_bioassay_table fetch first 10 rows only"
	sql = "SELECT * FROM temp_bioassay_table"
	df = sqlio.read_sql_query(sql, conn)

	return df.copy()

if __name__ == '__main__':
	db_user = sys.argv[1]
	db_password = sys.argv[2]
	df = chembl_activity_target(db_user, db_password)
	# df.to_csv('bioassay_table.csv')