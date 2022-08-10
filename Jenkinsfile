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
        PACKAGE_NAME = us4us.getPackageName(env, "${env.JOB_NAME}")
        PACKAGE_DIR = us4us.getUs4usJenkinsVariable(env, "PACKAGE_DIR")
        BUILD_TYPE = us4us.getBuildType(env)
        MISC_OPTIONS = us4us.getUs4usJenkinsVariable(env, "ARRUS_MISC_OPTIONS")
        US4R_API_RELEASE_DIR = us4us.getUs4rApiReleaseDir(env)
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
                    ${env.MISC_OPTIONS}
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
                        src_artifact='${env.RELEASE_DIR}/${env.JOB_NAME}/LICENSE;${env.RELEASE_DIR}/${env.JOB_NAME}/THIRD_PARTY_LICENSES;${env.RELEASE_DIR}/${env.JOB_NAME}/lib64;${env.RELEASE_DIR}/${env.JOB_NAME}/include;${env.RELEASE_DIR}/${env.JOB_NAME}/docs/arrus-cpp.pdf' \
                        dst_dir='${env.PACKAGE_DIR}/${env.JOB_NAME}'  \
                        dst_artifact='${env.PACKAGE_NAME}_cpp'
                   """
            }
        }
//         stage('Publish') {
//             steps {
//                   withCredentials([string(credentialsId: 'us4us-dev-github-token', variable: 'token')]){
//                   sh """pydevops --stage publish \
//                       --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
//                       ${DOCKER_DIRS} \
//                       ${SSH_DIRS} \
//                       --options \
//                       /publish/token='$token'
//                      """
//                 }
//             }
//         }
    }
}

def getBuildName(build) {
    wrap([$class: 'BuildUser']) {
        return "${env.PLATFORM} build #${build.id}, issued by: ${env.BUILD_USER_ID}, ${us4us.getCurrentDateTime()}";
    }
}