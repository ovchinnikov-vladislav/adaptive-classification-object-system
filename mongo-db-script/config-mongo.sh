echo "Creating mongo users..."
mongo -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --eval "db = db.getSiblingDB('stat'); db.createUser({user: 'stat', pwd: 'stat', roles: [{role: 'readWrite', db: 'stat'}]})"
echo "Mongo users created."