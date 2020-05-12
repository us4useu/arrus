pipeline {
    agent any

    stages {
        stage("Build dependencies") {
            steps {
                echo 'Building dependencies ...'
                build "us4r/${getBranchName()}"
            }
        }
        stage('Build') {
            steps {
                echo 'Building ...'
                sh "python '${env.WORKSPACE}/scripts/cfg_build.py' --source_dir '${env.WORKSPACE}' --us4r_dir '${env.US4R_INSTALL_DIR}/${getBranchName()}' --targets py matlab docs --run_targets tests"
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
    }
}

def getBranchName() {
    return env.BRANCH_NAME == "master" ? "master" : "develop";
}