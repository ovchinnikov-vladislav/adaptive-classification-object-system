echo
echo 'REMOVING ALL SERVICES'
docker rm $(docker ps -a -q)
echo
echo 'REMOVING DOCKER IMAGE services/discovery'
docker rmi services/discovery
echo 'DOCKERIZING discovery-service'
cd services/discovery-service
docker build --build-arg JAR_FILE=build/libs/*.jar -t services/discovery .
cd ../..
echo
echo 'REMOVING DOCKER IMAGE services/gateway'
docker rmi services/gateway
echo 'DOCKERIZING gateway-service'
cd services/gateway-service
docker build --build-arg JAR_FILE=build/libs/*.jar -t services/gateway .
cd ../..
echo
echo 'REMOVING DOCKER IMAGE services/result'
docker rmi services/result
echo 'DOCKERIZING result-service'
cd services/result-service
docker build --build-arg JAR_FILE=build/libs/*.jar -t services/result .
cd ../..
echo
echo 'REMOVING DOCKER IMAGE services/session'
docker rmi services/session
echo 'DOCKERIZING session-service'
cd services/session-service
docker build --build-arg JAR_FILE=build/libs/*.jar -t services/session .
cd ../..
echo
echo 'REMOVING DOCKER IMAGE services/task-executor'
docker rmi services/task-executor
echo 'DOCKERIZING task-executor-service'
cd services/task-executor-service
docker build --build-arg JAR_FILE=build/libs/*.jar -t services/task-executor .
cd ../..
echo
echo 'REMOVING DOCKER IMAGE services/task'
docker rmi services/task
echo 'DOCKERIZING task-service'
cd services/task-service
docker build --build-arg JAR_FILE=build/libs/*.jar -t services/task .
cd ../..
echo
echo 'REMOVING DOCKER IMAGE services/test'
docker rmi services/test
echo 'DOCKERIZING test-service'
cd services/test-service
docker build --build-arg JAR_FILE=build/libs/*.jar -t services/test .
cd ../..
echo
echo 'REMOVING DOCKER IMAGE services/front-end'
docker rmi services/front-end
echo 'DOCKERIZING front-end'
cd services/front-end
docker build --build-arg JAR_FILE=build/libs/*.jar -t services/front-end .
cd ../..
echo
echo 'START PUSH IMAGES TO DOCKER HUB'
docker login --username=vee97 --password=RabbitVlad1997
docker tag services/discovery vee97/discovery
echo 'PUSH vee97/discovery'
docker push vee97/discovery
docker tag services/gateway vee97/gateway
echo
echo 'PUSH vee97/gateway'
docker push vee97/gateway
docker tag services/result vee97/result
echo
echo 'PUSH vee97/result'
docker push vee97/result
docker tag services/session vee97/session
echo
echo 'PUSH vee97/session'
docker push vee97/session
docker tag services/task-executor vee97/task-executor
echo
echo 'PUSH vee97/task-executor'
docker push vee97/task-executor
docker tag services/task vee97/task
echo
echo 'PUSH vee97/task'
docker push vee97/task
docker tag services/test vee97/test
echo
echo 'PUSH vee97/test'
docker push vee97/test
docker tag services/front-end vee97/front-end
echo
echo 'PUSH vee97/front-end'
docker push vee97/front-end
echo

