This example shows how to integrate Flask-Login authentication with Flask-Admin using the SQLAlchemy backend.

To run this example:

1. Clone the repository::

     git clone https://github.com/mrjoes/flask-admin.git
     cd flask-admin

2. Create and activate a virtual environment::

     virtualenv env
     source env/bin/activate

3. Install requirements::

     pip install -r 'examples/auth/requirements.txt'

4. Run the application::

     python examples/auth/app.py

The first time you run this example, a sample sqlite database gets populated automatically. To suppress this behaviour,
comment the following lines in app.py:::

     if not os.path.exists(database_path):
         build_sample_db()
