pipeline {
    agent any

    parameters {
        booleanParam(name: 'PUBLISH_DOCS', defaultValue: false, description: 'Turns on publishing arrus docs on the documentation server.')
    }

    options {
        timeout(time: 1, unit: 'HOURS')
        buildDiscarder(logRotator(daysToKeepStr: '14'))
    }

    stages {
        stage("Build dependencies") {
            steps {
                echo 'Building dependencies ...'
                build "us4r/${getBranchName()}"
            }
        }
        stage('Configure') {
            steps {
                echo 'Configuring build ...'
                sh "python '${env.WORKSPACE}/scripts/cfg_build.py' --source_dir '${env.WORKSPACE}' --us4r_dir '${env.US4R_INSTALL_DIR}/${getBranchName()}' --targets py matlab docs --run_targets tests"
            }
        }
        stage('Build') {
            steps {
                echo 'Building ...'
                sh "python '${env.WORKSPACE}/scripts/build.py' --source_dir'=${env.WORKSPACE}'"
            }
        }
        stage('Test') {
            environment {
                Path = "${env.US4R_INSTALL_DIR}/${getBranchName()}/lib64;${env.Path}"
            }
            steps {
                echo 'Testing ...'
                sh "python '${env.WORKSPACE}/scripts/test.py' --source_dir='${env.WORKSPACE}'"
            }
        }
        stage('Install') {
            steps {
                echo 'Installing ...'
                sh "python '${env.WORKSPACE}/scripts/install.py' --source_dir='${env.WORKSPACE}' --install_dir='${env.ARRUS_INSTALL_DIR}/${env.BRANCH_NAME}'"
            }
        }
        stage('Publish docs') {
            when{
                environment name: 'PUBLISH_DOCS', value: 'true'
                anyOf {
                    branch 'master'
                    branch 'develop'
                }
            }
            steps {
                echo "Publishing docs ..."
                withCredentials([usernamePassword(credentialsId: '00c79f7e-f299-4ec6-959d-9d09785891f5', usernameVariable: 'username', passwordVariable: 'password')]){
                {
                    sh("git clone https://$username:$password@github.com/us4useu/arrus-public.git")
                }
            }
        }
    }
}

def getBranchName() {
    return env.BRANCH_NAME == "master" ? "master" : "develop";
}