import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ Create a database connection to the SQLite database specified by db_file """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    """ Create a table from the create_table_sql statement """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def setup_database(db_file="database.sqlite"):
    sql_create_feedback_table = """ CREATE TABLE IF NOT EXISTS feedback (
                                        id integer PRIMARY KEY,
                                        filename text NOT NULL,
                                        correct_label text,
                                        predicted_label text,
                                        feedback text
                                    ); """

    # Create a database connection
    conn = create_connection(db_file)

    # Create table
    if conn is not None:
        create_table(conn, sql_create_feedback_table)
    else:
        print("Error! Cannot create the database connection.")


def add_feedback(conn, filename, correct_label, predicted_label, feedback):
    try:
        with conn:
            sql = ''' INSERT INTO feedback(filename, correct_label, predicted_label, feedback)
                      VALUES(?,?,?,?) '''
            cur = conn.cursor()
            cur.execute(sql, (filename, correct_label, predicted_label, feedback))
            conn.commit()

            print(f'Feedback added to the database: {filename}, {correct_label}, {predicted_label}, {feedback}')
    except sqlite3.OperationalError:
        setup_database()


if __name__ == '__main__':
    setup_database()
