pipeline {
    agent any
    stages {

	stage('build and test') {
	    steps {
		echo 'STAGE BUILD AND TEST'
		sh './gradlew clean build bootJar'
	    }
        }

	stage('dockerizing') {
	    steps {
		echo 'STAGE DOCKERIZING'
		sh 'sudo chmod +x start_docker.sh'
	    	sh 'sudo ./start_docker.sh'
		sh 'sudo chmod +x create_service_images_docker.sh'
	    	sh 'sudo ./create_service_images_docker.sh'
    	    }
        }

	stage('deploy') {
	    steps {
		echo 'STAGE DEPLOY'
		sh 'sudo chmod +x deploy.sh'
	    	sh 'sudo ./deploy.sh'
	    }	
	}
    }
}
