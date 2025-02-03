from sqlalchemy import create_engine

engine = create_engine('postgresql://root:root@pgdatabase:5432/testdb')

def save_to_db(cleaned):
    if(engine.connect()):
        print('Connected to Database')
        try:
            print('Writing cleaned dataset to database')
            cleaned.to_sql('fintech_data_MET_52_23665_clean', con=engine, if_exists='append')
            print('Done writing to database')
        except ValueError as vx:
            print('Cleaned Table already exists.')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')
def save_lookup(cleaned,):
    if(engine.connect()):
        print('Connected to Database')
        try:
            print('Writing lookup to database')
            cleaned.to_sql('lookup_table_MET_52_23665_clean', con=engine, if_exists='replace')
            print('Done writing to database')
        except ValueError as vx:
            print('Cleaned Table already exists.')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')