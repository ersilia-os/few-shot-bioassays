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
	"""
	Filtering:
		We want to isolate columns
			from target_dictionary:
				target_type (filtered on organism)
				pref_name AS target_pref_name
			from assays:
				assay_id
				organism
				confidence_score
			from organism_class:
				organism_taxonomy_l1 (filtered on bacteria, viruses, fungi)
				organism_taxonomy_l2,
				organism_taxonomy_l3
			from compound_structures:
				canonical_smiles
			from activities:
				pchembl_value
		We join tables as:
			target_dictionary and assays based on target_id (tid)
			assays and organism_class based on taxonomy_id (assay_tax_id/tax_id)
			assays and activities based on assay_id
			activities and molecule_dictionary based on molregno
			molecule_dictionary and compound_structures based on molregno
			
		We want to filter data:
			with pchembl_value > 0
			with target_type = 'organism'
			with organism_taxonomy_l1 IN ['bacteria', 'viruses', 'fungi']	
	"""


	query = """
		CREATE TEMP TABLE temp_bioassay_table AS
		SELECT 
			a.assay_id,
			td.target_type,
			td.pref_name AS target_pref_name,
			a.assay_organism,
			a.confidence_score,
			oct.l1 AS organism_taxonomy_l1,
			oct.l2 AS organism_taxonomy_l2,
			oct.l3 AS organism_taxonomy_l3,
			cnd_s.canonical_smiles AS smiles, 
			act.pchembl_value
		FROM target_dictionary td
		INNER JOIN assays a ON td.tid = a.tid
		INNER JOIN organism_class oct ON a.assay_tax_id = oct.tax_id
		INNER JOIN activities act ON a.assay_id = act.assay_id
		INNER JOIN molecule_dictionary md ON act.molregno = md.molregno
		INNER JOIN compound_structures cnd_s ON md.molregno = cnd_s.molregno
		WHERE act.pchembl_value > 0
		AND oct.l1 IN ('Bacteria', 'Viruses', 'Fungi')
		AND td.target_type = 'ORGANISM'
		"""
	cursor.execute(query)

	cursor.execute("SELECT count(*) FROM temp_bioassay_table")
	print(f'{cursor.fetchone()[0]} rows extracted from Chembl')
	cursor.execute("SELECT COUNT(DISTINCT assay_id) FROM temp_bioassay_table")
	print(f'{cursor.fetchone()[0]} unique assays extracted from Chembl')


	sql = "SELECT * FROM temp_bioassay_table"
	df = sqlio.read_sql_query(sql, conn)

	return df.copy()

if __name__ == '__main__':
	db_user = sys.argv[1]
	db_password = sys.argv[2]
	df = chembl_activity_target(db_user, db_password)
	df.to_csv('bioassay_table_filtered.csv')