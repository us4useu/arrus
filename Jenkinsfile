@Library("us4us-jenkins-shared-libraries@master") _;

pipeline {
    agent any

    environment {
        PLATFORM = us4us.getPlatformName(env)
        BUILD_ENV_ADDRESS = us4us.getUs4usJenkinsVariable(env, "BUILD_ENV_ADDRESS")
        DOCKER_OPTIONS = us4us.getUs4usJenkinsVariable(env, "ARRUS_DOCKER_OPTIONS") // Deprecated
        DOCKER_OPTIONSv2 = us4us.getUs4usJenkinsVariable(env, "ARRUS_DOCKER_OPTIONSv2")  // Docker options for ARRUS >= 0.9.0.
        DOCKER_DIRS = us4us.getRemoteDirs(env, "docker", "DOCKER_BUILD_ROOT")
        SSH_DIRS = us4us.getRemoteDirs(env, "ssh", "SSH_BUILD_ROOT")
        TARGET_WORKSPACE_DIR = us4us.getTargetWorkspaceDir(env, "DOCKER_BUILD_ROOT", "SSH_BUILD_ROOT")
        CONAN_HOME_DIR = us4us.getUs4usJenkinsVariable(env, "CONAN_HOME_DIR")
        CONAN_PROFILE_FILE = us4us.getConanProfileFile(env)
        RELEASE_DIR = us4us.getUs4usJenkinsVariable(env, "RELEASE_DIR")
        CPP_PACKAGE_NAME = us4us.getPackageName(env, "${env.JOB_NAME}", "cpp")
        MATLAB_PACKAGE_NAME = us4us.getPackageName(env, "${env.JOB_NAME}", "matlab")
        PACKAGE_DIR = us4us.getUs4usJenkinsVariable(env, "PACKAGE_DIR")
        BUILD_TYPE = us4us.getBuildType(env)
        MISC_OPTIONS = us4us.getUs4usJenkinsVariable(env, "ARRUS_MISC_OPTIONS")
        US4R_API_RELEASE_DIR = us4us.getUs4rApiReleaseDir(env)
        IS_ARRUS_WHL_SUFFIX = us4us.isArrusSuffixWhl(env)
    }

     parameters {
        booleanParam(name: 'PUBLISH_DOCS', defaultValue: false, description: 'Publish arrus docs on the documentation server. CHECKING THIS ONE WILL UPDATE ARRUS DOCS')
        booleanParam(name: 'PUBLISH_CPP', defaultValue: false, description: 'Publish arrus C++ API package.')
        booleanParam(name: 'PUBLISH_PY', defaultValue: false, description: 'Publish arrus Python package.')
        booleanParam(name: 'PUBLISH_MATLAB', defaultValue: false, description: 'Publish arrus MATLAB package.')
        choice(name: 'PY_VERSION', choices: ['3.8', '3.9', '3.10'], description: 'Python version to use.')
     }

    stages {
        stage('Configure') {
            steps {
                sh """
                   pydevops --clean --stage cfg \
                    --host '${env.BUILD_ENV_ADDRESS}' \
                    ${getDockerOptionsForTemplate(env.DOCKER_OPTIONSv2)}  \
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
                    /cfg/cmake/DARRUS_APPEND_VERSION_SUFFIX_DATE=${IS_ARRUS_WHL_SUFFIX} \
                    /cfg/DARRUS_PY_VERSION=${params.PY_VERSION} \
                    ${getPythonExecutableParameter(env, ${params.PY_VERSION})}
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
        stage('PackageMatlab') {
             steps {
                 sh """pydevops --stage package_matlab \
                       --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                       ${DOCKER_DIRS} \
                       ${SSH_DIRS} \
                       --options \
                       release_name='${env.BRANCH_NAME}' \
                       src_artifact='${env.RELEASE_DIR}/${env.JOB_NAME}/matlab' \
                       dst_dir='${env.PACKAGE_DIR}/${env.JOB_NAME}'  \
                       dst_artifact='${env.MATLAB_PACKAGE_NAME}'
                    """
             }
         }
        stage('PublishCpp') {
            when{
                environment name: 'PUBLISH_CPP', value: 'true'
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
                      repository_name='us4useu/arrus' \
                      description='${getBuildName(currentBuild)} (C++)'
                     """
                }
            }
        }
        stage('PublishPython') {
            when{
                environment name: 'PUBLISH_PY', value: 'true'
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
                     src_artifact='${env.RELEASE_DIR}/${env.JOB_NAME}/python/${getArrusWhlNamePattern()}' \
                     dst_artifact='__same__' \
                     repository_name='us4useu/arrus' \
                     description='${getBuildName(currentBuild)} (Python)'
                     """
                }
            }
        }
        stage('PublishMatlab') {
            when{
                environment name: 'PUBLISH_MATLAB', value: 'true'
            }
            steps {
                  withCredentials([string(credentialsId: 'us4us-dev-github-token', variable: 'token')]){
                  sh """pydevops --stage publish_matlab \
                     --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                     ${DOCKER_DIRS} \
                     ${SSH_DIRS} \
                     --options \
                     token='$token' \
                     release_name='${env.BRANCH_NAME}' \
                     src_artifact='${env.PACKAGE_DIR}/${env.JOB_NAME}/${env.MATLAB_PACKAGE_NAME}*' \
                     dst_artifact='__same__' \
                     repository_name='us4useu/arrus' \
                     description='${getBuildName(currentBuild)} (MATLAB)'
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
                      repository='https://$username:$password@github.com/us4useu/arrus-docs.git' \
                      commit_msg='Updated docs, ${getBuildName(currentBuild)}'
                      """
                 }
             }
         }


    }
}

def getArrusWhlNamePattern() {
    pythonVersion = "cp${params.PY_VERSION}".replace(".", "");
    if(us4us.isPrerelease("${env.BRANCH_NAME}")) {

        return "arrus*${us4us.getTimestamp()}*${pythonVersion}*.whl";
    }
    else {
        return "arrus*${pythonVersion}*.whl";
    }
}

def getBuildName(build) {
    wrap([$class: 'BuildUser']) {
        return "${env.PLATFORM} build ${build.id}, issued by: ${env.BUILD_USER_ID}, ${us4us.getCurrentDateTime()}";
    }
}

def getDockerOptionsForTemplate(dockerOptionsTemplate) {
    return dockerOptionsTemplate.replace("%%PY_VERSION%%", "${params.PY_VERSION}");
}

def getPythonExecutableParameter(env, pythonVersion) {
    print "Calling Python Executable Parameter!";
    var sanitizedPyVersion = pythonVersion.replace(".", "");
    var pythonExecutablePath = us4us.getUs4usJenkinsVariable(env, "PYTHON_EXECUTABLE_${sanitizedPyVersion}");
    if(pythonExecutablePath != null && !pythonExecutablePath.trim().isEmpty()) {
        return "/cfg/DPYTHON_EXECUTABLE=${pythonExecutablePath}";
    }
    else {
        return "";
    }
}