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
		JOIN assays a ON td.tid = a.tid
		JOIN activities act ON a.assay_id = act.assay_id
		JOIN molecule_dictionary md ON act.molregno = md.molregno
		JOIN compound_structures cnd_s ON md.molregno = cnd_s.molregno
		"""
	cursor.execute(query)

	cursor.execute("SELECT count(*) FROM temp_bioassay_table")
	print(f'{cursor.fetchone()[0]} rows extracted from Chembl')

	sql = "SELECT * FROM temp_bioassay_table"
	df = sqlio.read_sql_query(sql, conn)

	return df.copy()

if __name__ == '__main__':
	db_user = "chembl_user"
	db_password = "aaa"
	df = chembl_activity_target(db_user, db_password)

	# TODO: convert the df to CSV / decide on something that'll make it easy for us to run pytorch on this