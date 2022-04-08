import pyodbc as py

def connect_to_sql_server(l_sql_server):
    sql_server_connection = pyodbc.connect('Driver={SQL Server};'
                                           'Server=' + l_sql_server + ';'
                                           'Database=odyssey_broker_quickfeeUS_REPORTING;'
                                           'Trusted_Connection=yes;')

    if sql_server_connection:
        print ('connected to server')

    return sql_server_connection

def export_employee_image(sl_connection, alias, l_name,l_file):
    sql_stmt = "select CONVERT(varbinary(max),SignatureImage) FROM users up WHERE Intermediary = '''TEST''' "
    cursor = sl_connection.cursor()
    cursor.execute(sql_stmt)

sql_connection = connect_to_sql_server(('10.40.0.4'))
if sql_connection:
    export_employee_image(sql_connection,'TEST','TEST.png','/Documents/Github/datasciencecoursera/')