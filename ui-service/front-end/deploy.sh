echo
echo 'START DOCKER-MACHINE managervm'
docker-machine start managervm
echo
echo 'START DOCKER-MACHINE worker1vm'
docker-machine start worker1vm
echo
echo 'START DOCKER-MACHINE worker2vm'
docker-machine start worker2vm
echo
echo 'START DOCKER-MACHINE worker3vm'
docker-machine start worker3vm
echo
docker-machine ls
docker-machine ssh managervm "docker node ls"
docker-machine scp docker-compose.yml managervm:~
docker-machine ssh managervm "mkdir pg-init-scripts"
docker-machine ssh worker1vm "mkdir pg-init-scripts"
docker-machine ssh worker2vm "mkdir pg-init-scripts"
docker-machine ssh worker3vm "mkdir pg-init-scripts"
docker-machine scp pg-init-scripts/create-multiple-postgresql-databases.sh managervm:~/pg-init-scripts
docker-machine scp pg-init-scripts/create-multiple-postgresql-databases.sh worker1vm:~/pg-init-scripts
docker-machine scp pg-init-scripts/create-multiple-postgresql-databases.sh worker2vm:~/pg-init-scripts
docker-machine scp pg-init-scripts/create-multiple-postgresql-databases.sh worker3vm:~/pg-init-scripts
echo
echo 'REMOVING STACK services FOR CLUSTER'
docker-machine ssh managervm "docker stack rm services"
echo
echo 'REMOVING NETWORK webnet FOR CLUSTER'
docker-machine ssh managervm "docker network rm webnet"
echo
echo 'CREATING NETWORK webnet FOR CLUSTER'
docker-machine ssh managervm "docker network create --driver overlay webnet"
echo
echo 'DEPLOY services FOR CLUSTER'
docker-machine ssh managervm "docker stack deploy -c docker-compose.yml services"
echo
docker-machine ssh managervm "docker stack ps services"
