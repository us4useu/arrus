pipeline {
    agent any

    stages {
        stage("Build dependencies") {
            when {
                expression {env.BRANCH_NAME != 'ref-57'}
            }
            steps {
                echo 'Building dependencies..'
                build 'us4r/ref-75'
            }
        }
        stage('Build') {
            steps {
                echo 'Building..'
                withCredentials() {
                }
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}