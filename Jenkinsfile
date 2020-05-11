pipeline {
    agent any

    stages {
        stage("Build dependencies (master)") {
            when {
                expression {env.BRANCH_NAME == 'master'}
            }
            steps {
                echo 'Building dependencies (master) ...'
                build 'us4r/master'
            }
        }
        stage("Build dependencies (develop)") {
            when {
                expression {env.BRANCH_NAME != 'master'}
            }
            steps {
                echo 'Building dependencies..'
                build 'us4r/develop'
            }
        }
        stage('Build') {
            steps {
                echo 'Building ...'
                sh '''
                python '${env.BRANCH_NAME}/scripts/'
                '''
            }
        }
        stage('Test') {
            steps {
                echo 'Testing ...'
            }
        }
        stage('Install') {
            steps {
                echo 'Deploying ...'
            }
        }
        stage('Install') {
            parallel {
            }
        }
    }
}