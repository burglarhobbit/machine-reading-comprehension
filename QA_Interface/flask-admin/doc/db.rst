Database backends
=================

The purpose of Flask-Admin is to help you manage your data. For this, it needs some database backend in order to be
able to access that data in the first place. At present, there are five different backends for you to choose
from, depending on which database you would like to use for your application.

.. toctree::
   :maxdepth: 2

   db_sqla
   db_geoa
   db_mongoengine
   db_peewee
   db_pymongo

If you don't know where to start, but you're familiar with relational databases, then you should probably look at using
`SQLAlchemy`_. It is a full-featured toolkit, with support for SQLite, PostgreSQL, MySQL,
Oracle and MS-SQL amongst others. It really comes into its own once you have lots of data, and a fair amount of
relations between your data models. If you want to track spatial data like latitude/longitude
points, you should look into `GeoAlchemy`_, as well.

If you're looking for something simpler, or your data models are reasonably self-contained, then
`MongoEngine`_ could be a better option. It is a python wrapper around the popular
*NoSQL* database called `MongoDB`_.

Of course, if you feel that there's an awesome database wrapper that is missing from the list above, we'd greatly
appreciate it if you could write the plugin for it and submit it as a pull request. A special section of these docs
are dedicated to helping you through this process. See :doc:`model_guidelines`.

.. _SQLAlchemy: http://www.sqlalchemy.org/
.. _GeoAlchemy: http://geoalchemy-2.readthedocs.org/
.. _MongoEngine: http://mongoengine.org/
.. _MongoDB: http://www.mongodb.org/

