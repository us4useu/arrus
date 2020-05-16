pipeline {
    agent any

    parameters {
        booleanParam(name: 'PUBLISH_DOCS', defaultValue: false, description: 'Turns on publishing arrus docs on the documentation server. CHECKING THIS ONE WILL UPDATE ARRUS DOCS')
        booleanParam(name: 'PUBLISH_PACKAGE', defaultValue: false, description: 'Turns on publishing arrus package with binary release on the github server. CHECKING THIS ONE WILL UPDATE ARRUS RELEASE')
    }

    options {
        timeout(time: 1, unit: 'HOURS')
        buildDiscarder(logRotator(daysToKeepStr: '14'))
    }

    stages {
        stage('Prepare build environment') {
            steps {
                script {
                    currentBuild.displayName = getBuildName(currentBuild)
                }
                sh "git clean -fdx"
            }
        }
        stage("Build dependencies") {
            steps {
                echo 'Building dependencies ...'
                build "us4r/${getBranchName()}"
            }
        }
        stage('Configure') {
            steps {
                echo 'Configuring build ...'
                sh "python '${env.WORKSPACE}/scripts/cfg_build.py' --source_dir '${env.WORKSPACE}' --us4r_dir '${env.US4R_INSTALL_DIR}/${getBranchName()}' --targets py matlab docs --run_targets tests --options ARRUS_EMBED_DEPS=ON"
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
        stage('Publish package') {
            when{
                environment name: 'PUBLISH_PACKAGE', value: 'true'
                anyOf {
                    branch 'master'
                    branch 'develop'
                }
            }
            steps {
                echo "Publishing release package ..."
                withCredentials([string(credentialsId: 'us4us-support-github-token', variable: 'token')]){
                    sh "python '${env.WORKSPACE}/scripts/publish_releases.py' --install_dir='${env.ARRUS_INSTALL_DIR}/${env.BRANCH_NAME}' --repository 'us4useu/arrus-public' --src_branch_name ${env.BRANCH_NAME} --token $token --build_id '${getBuildName(currentBuild)}'"
                }
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
                withCredentials([usernamePassword(credentialsId: 'us4us-support-github-credentials', usernameVariable: 'username', passwordVariable: 'password')]){
                    sh "python '${env.WORKSPACE}/scripts/publish_docs.py' --install_dir='${env.ARRUS_INSTALL_DIR}/${env.BRANCH_NAME}' --repository 'https://$username:$password@github.com/us4useu/arrus-public.git' --src_branch_name ${env.BRANCH_NAME} --build_id '${getBuildName(currentBuild)}'"
                }
            }
        }
    }
    post {
        failure {
            script {
                if(env.BRANCH_NAME == "master" || env.BRANCH_NAME == "develop") {
                    emailext(body:    "Check console output at $BUILD_URL to view the results.",
                             to:      "${env.ARRUS_DEVELOPER_EMAIL}",
                             subject: "Build failed in Jenkins: $JOB_NAME")
                }
            }
        }
        unstable {
            script {
                if(env.BRANCH_NAME == "master" || env.BRANCH_NAME == "develop") {
                    emailext(body:    "Check console output at $BUILD_URL to view the results.",
                             to:      "${env.ARRUS_DEVELOPER_EMAIL}",
                             subject: "Unstable build in Jenkins: $JOB_NAME")
                }
            }
        }
        changed {
            script {
                if(env.BRANCH_NAME == "master" || env.BRANCH_NAME == "develop") {
                    emailext(body:    "Check console output at $BUILD_URL to view the results.",
                             to:      "${env.ARRUS_DEVELOPER_EMAIL}",
                             subject: "Jenkins build is back to normal: $JOB_NAME")
                }
            }
        }
    }
}

def getBranchName() {
    return env.BRANCH_NAME == "master" ? "master" : "develop";
}

def getBuildName(build) {
    wrap([$class: 'BuildUser']) {
        return "#${build.id} (${env.BUILD_USER_ID})";
    }
}