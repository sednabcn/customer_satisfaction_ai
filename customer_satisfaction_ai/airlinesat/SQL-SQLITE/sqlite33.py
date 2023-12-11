import sqlite3

# We create the connection to the example.db database
# If the database file is not there it is created.
con = sqlite3.connect('consuexp01.db')

# Create a table
statement="SELECT * FROM airlinessat LIMIT 5;"
#"CREATE TABLE revenues (date text, amount float, source text)")
cursor=con.execute(statement)

# results
rows=cursor.fetchall()
for row in rows:
    print(row)
# We commit the changes
con.commit()
