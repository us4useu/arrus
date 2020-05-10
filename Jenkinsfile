pipeline {
    agent any

    stages {
        stage("Build dependencies") {
            when {
                expression {env.BRANCH_NAME != 'master'}
            }
            steps {
                echo 'Building dependencies..'
                build 'us4r/ref-75'
            }
        }
        stage('Build') {
            steps {
                echo 'Building..'
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