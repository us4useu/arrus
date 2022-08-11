@Library("us4us-jenkins-shared-libraries@master") _;

pipeline {
    agent any

    environment {
        PLATFORM = us4us.getPlatformName(env)
        BUILD_ENV_ADDRESS = us4us.getUs4usJenkinsVariable(env, "BUILD_ENV_ADDRESS")
        DOCKER_OPTIONS = us4us.getUs4usJenkinsVariable(env, "ARRUS_DOCKER_OPTIONS")
        DOCKER_DIRS = us4us.getRemoteDirs(env, "docker", "DOCKER_BUILD_ROOT")
        SSH_DIRS = us4us.getRemoteDirs(env, "ssh", "SSH_BUILD_ROOT")
        TARGET_WORKSPACE_DIR = us4us.getTargetWorkspaceDir(env, "DOCKER_BUILD_ROOT", "SSH_BUILD_ROOT")
        CONAN_HOME_DIR = us4us.getUs4usJenkinsVariable(env, "CONAN_HOME_DIR")
        CONAN_PROFILE_FILE = us4us.getConanProfileFile(env)
        RELEASE_DIR = us4us.getUs4usJenkinsVariable(env, "RELEASE_DIR")
        CPP_PACKAGE_NAME = us4us.getPackageName(env, "${env.JOB_NAME}", "cpp")
        PACKAGE_DIR = us4us.getUs4usJenkinsVariable(env, "PACKAGE_DIR")
        BUILD_TYPE = us4us.getBuildType(env)
        MISC_OPTIONS = us4us.getUs4usJenkinsVariable(env, "ARRUS_MISC_OPTIONS")
        US4R_API_RELEASE_DIR = us4us.getUs4rApiReleaseDir(env)
        IS_ARRUS_WHL_SUFFIX = us4us.isArrusSuffixWhl(env)
    }

     parameters {
        booleanParam(name: 'PUBLISH_DOCS', defaultValue: false, description: 'Turns on publishing arrus docs on the documentation server. CHECKING THIS ONE WILL UPDATE ARRUS DOCS')
        booleanParam(name: 'PUBLISH_PACKAGE', defaultValue: false, description: 'Turns on publishing arrus package with binary release on the github server. CHECKING THIS ONE WILL UPDATE ARRUS RELEASE')
     }

    stages {
        stage('Configure') {
            steps {
                sh """
                   pydevops --clean --stage cfg \
                    --host '${env.BUILD_ENV_ADDRESS}' \
                    ${env.DOCKER_OPTIONS}  \
                    --src_dir '${env.WORKSPACE}' --build_dir '${env.WORKSPACE}/build' \
                    ${env.DOCKER_DIRS} \
                    ${env.SSH_DIRS} \
                    --options \
                    build_type='${env.BUILD_TYPE}' \
                    us4r_api_release_dir='${env.US4R_API_RELEASE_DIR}' \
                    /cfg/conan/conan_home='${env.CONAN_HOME_DIR}' \
                    /cfg/conan/profile='${env.TARGET_WORKSPACE_DIR}/.conan/${env.CONAN_PROFILE_FILE}' \
                    /install/prefix='${env.RELEASE_DIR}/${env.JOB_NAME}' \
                    ${env.MISC_OPTIONS} \
                    /cfg/cmake/DARRUS_APPEND_VERSION_SUFFIX_DATE=${IS_ARRUS_WHL_SUFFIX}
                    """
            }
        }
        stage('Build') {
            steps {
                sh """pydevops --stage build \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${env.DOCKER_DIRS} \
                      ${env.SSH_DIRS}
                   """
            }
        }
        stage('Test') {
            steps {
                sh """pydevops --stage test \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${env.DOCKER_DIRS} \
                      ${env.SSH_DIRS}
                   """
            }
        }
        stage('Install') {
            steps {
                sh """pydevops --stage install \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${DOCKER_DIRS} \
                      ${SSH_DIRS}
                   """
            }
        }
        stage('PackageCpp') {
            steps {
                sh """pydevops --stage package_cpp \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${DOCKER_DIRS} \
                      ${SSH_DIRS} \
                      --options \
                      release_name='${env.BRANCH_NAME}' \
                      src_artifact='${env.RELEASE_DIR}/${env.JOB_NAME}/LICENSE;${env.RELEASE_DIR}/${env.JOB_NAME}/THIRD_PARTY_LICENSES;${env.RELEASE_DIR}/${env.JOB_NAME}/lib64;${env.RELEASE_DIR}/${env.JOB_NAME}/include;${env.RELEASE_DIR}/${env.JOB_NAME}/docs/arrus-cpp.pdf;${env.RELEASE_DIR}/${env.JOB_NAME}/examples' \
                      dst_dir='${env.PACKAGE_DIR}/${env.JOB_NAME}'  \
                      dst_artifact='${env.CPP_PACKAGE_NAME}'
                   """
            }
        }
        stage('PublishCpp') {
            when{
                environment name: 'PUBLISH_PACKAGE', value: 'true'
            }
            steps {
                  withCredentials([string(credentialsId: 'us4us-dev-github-token', variable: 'token')]){
                  sh """pydevops --stage publish_cpp \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${DOCKER_DIRS} \
                      ${SSH_DIRS} \
                      --options \
                      token='$token' \
                      release_name='${env.BRANCH_NAME}' \
                      src_artifact='${env.PACKAGE_DIR}/${env.JOB_NAME}/${env.CPP_PACKAGE_NAME}*' \
                      dst_artifact='__same__' \
                      repository_name='pjarosik/arrus' \
                      description='${getBuildName(currentBuild)} (C++)'
                     """
                }
            }
        }
        stage('PublishPython') {
            when{
                environment name: 'PUBLISH_PACKAGE', value: 'true'
            }
            steps {
                  withCredentials([string(credentialsId: 'us4us-dev-github-token', variable: 'token')]){
                  sh """pydevops --stage publish_py \
                     --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                     ${DOCKER_DIRS} \
                     ${SSH_DIRS} \
                     --options \
                     token='$token' \
                     release_name='${env.BRANCH_NAME}' \
                     src_artifact='${env.RELEASE_DIR}/${env.JOB_NAME}/python/arrus*.whl' \
                     dst_artifact='__same__' \
                     repository_name='pjarosik/arrus' \
                     description='${getBuildName(currentBuild)} (Python)'
                     """
                }
            }
        }
        stage('PublishDocs') {
             when{
                 environment name: 'PUBLISH_DOCS', value: 'true'
             }
             steps {
                   withCredentials([usernamePassword(credentialsId: 'us4us-dev-github-credentials', usernameVariable: 'username', passwordVariable: 'password')]){
                   sh """pydevops --stage publish_docs \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${DOCKER_DIRS} \
                      ${SSH_DIRS} \
                      --options \
                      version='${env.BRANCH_NAME}' \
                      install_dir='${env.RELEASE_DIR}/${env.JOB_NAME}/' \
                      repository='https://$username:$password@github.com/pjarosik/arrus-docs.git' \
                      commit_msg='Updated docs, ${getBuildName(currentBuild)}'
                      """
                 }
             }
         }


    }
}

def getBuildName(build) {
    wrap([$class: 'BuildUser']) {
        return "${env.PLATFORM} build ${build.id}, issued by: ${env.BUILD_USER_ID}, ${us4us.getCurrentDateTime()}";
    }
}