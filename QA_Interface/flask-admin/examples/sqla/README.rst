SQLAlchemy model backend integration examples.

To run this example:

1. Clone the repository::

    git clone https://github.com/mrjoes/flask-admin.git
    cd flask-admin

2. Create and activate a virtual environment::

    virtualenv env
    source env/bin/activate

3. Install requirements::

    pip install -r 'examples/sqla/requirements.txt'

4. Run either of these applications::

    python examples/sqla/app.py
    python examples/sqla/app2.py

The first time you run this example, a sample sqlite database gets populated automatically. To suppress this behaviour,
comment the following lines in app.py:::

    if not os.path.exists(database_path):
        build_sample_db()
