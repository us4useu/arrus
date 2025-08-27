@Library("us4us-jenkins-shared-libraries@master") _;

pipeline {
    agent any

    parameters {
        booleanParam(name: 'RELEASE', defaultValue: false, description: 'Is this release? When set to true, the VERSION parameter is required.')
        string(name: 'VERSION', defaultValue: '', description: 'Release version number.')
        booleanParam(name: 'PUBLISH', defaultValue: false, description: 'Publish arrus on github server. When set to true, only the publish stages will be executed. When set to false, only the build stages will be executed. This will publish the latest build of the given release or branch. NOTE: the scope of publication can be limited using PUBLISH_PY/PUBLISH_CPP/PUBLISH_MATLAB')
        booleanParam(name: 'PUBLISH_PY', defaultValue: false, description: 'Publish Python package.')
        booleanParam(name: 'PUBLISH_MATLAB', defaultValue: false, description: 'Publish Matlab package.')
        booleanParam(name: 'PUBLISH_CPP', defaultValue: false, description: 'Publish Matlab package.')
        booleanParam(name: 'PUBLISH_DOCS', defaultValue: false, description: 'Publish ARRUS documentation (web).')
        choice(name: 'PY_VERSION', choices: ['3.8', '3.9', '3.10'], description: 'Python version to use.')
        booleanParam(name: 'SCM_ONLY', defaultValue: false, description: 'Perform SCM checkout only, in order to e.g. update parameters of the pipeline.')
     }

    environment {
        PROJECT_NAME = "arrus-test"
        PLATFORM = us4us.getPlatformName(env)
        BUILD_ENV_ADDRESS = us4us.getUs4usJenkinsVariable(env, "BUILD_ENV_ADDRESS")
        DOCKER_OPTIONS = us4us.getUs4usJenkinsVariable(env, "ARRUS_DOCKER_OPTIONS") // Deprecated
        DOCKER_OPTIONSv2 = us4us.getUs4usJenkinsVariable(env, "ARRUS_DOCKER_OPTIONSv2")  // Docker options for ARRUS >= 0.9.0.
        DOCKER_DIRS = us4us.getRemoteDirs(env, "docker", "DOCKER_BUILD_ROOT")
        SSH_DIRS = us4us.getRemoteDirs(env, "ssh", "SSH_BUILD_ROOT")
        TARGET_WORKSPACE_DIR = us4us.getTargetWorkspaceDir(env, "DOCKER_BUILD_ROOT", "SSH_BUILD_ROOT")
        TARGET_PRERELEASE_DIR = us4us.getTargetArtifactsDir(env, params, "${env.JOB_NAME}", false, "arrus-test")
        TARGET_RELEASE_DIR = us4us.getTargetArtifactsDir(env, params, "${env.JOB_NAME}", true, "arrus-test")
        CONAN_HOME_DIR = us4us.getUs4usJenkinsVariable(env, "CONAN_HOME_DIR")
        CONAN_PROFILE_FILE = us4us.getConanProfileFile(env)
        BUILD_TYPE = us4us.getBuildType(env)
        MISC_OPTIONS = us4us.getUs4usJenkinsVariable(env, "ARRUS_MISC_OPTIONS")
        IS_SCM_ONLY = isSCMOnly(params)
        INSTALL_DIR_PREFIX = "${TARGET_PRERELEASE_DIR}/unzipped"
    }

    stages {
        stage('Fetch tags') {
            when{
                expression { us4us.isReleaseBranch("${env.BRANCH_NAME}") }
            }
            steps {
                sh 'git fetch --prune --tags --force'
            }
        }
        stage('Skip Build?') {
            when {
                environment name: 'SCM_ONLY', value: 'true'
            }
            steps {
                script {
                    currentBuild.result = 'ABORTED'
                    error("Skipping the Job to update the build info")
                }
            }
        }
        stage("Validate parameters") {
            steps {
                script {
                    us4us.validateParameters(env, params);
                }
            }
        }

        // ------------------------------------------ BUILD STAGES.

        stage('Configure') {
            // It is always required, even if it is just publishing -- just to handle properly the PublishGithub stages.
            steps {
                script {
                    env.CPP_PACKAGE_NAME = us4us.getPackageNameV2(env, params, "${env.JOB_NAME}", "cpp");
                    env.MATLAB_PACKAGE_NAME = us4us.getPackageNameV2(env, params, "${env.JOB_NAME}", "matlab");
                    // Release name: version number if this stable release, or pre-release if this is dev.
                    def releaseName = us4us.getReleaseName(env, params);
                    env.RELEASE_NAME = releaseName;
                    // In case we are not performing the official release
                    // e.g. we are building -dev package, publish the packages
                    // from the pre-release dir.
                    // This one is for C++ and MATLAB (the pre-release artifacts are in the .../pre-release/directory).
                    def githubSourceArtifactPath = us4us.isPrereleaseV2(params) ? "${TARGET_PRERELEASE_DIR}": "${TARGET_RELEASE_DIR}";
                    env.GITHUB_SOURCE_ARTIFACT_PATH = githubSourceArtifactPath;
                    // This one is for Python and docs (the pre-release artifacts are in the .../pre-release/unzipped/{RELEASE_NAME} directory).
                    def pyArtifactPath = us4us.isPrereleaseV2(params) ? "${TARGET_PRERELEASE_DIR}/unzipped/${releaseName}/python": "${TARGET_RELEASE_DIR}";
                    env.GITHUB_PY_ARTIFACT_PATH = pyArtifactPath;
                    // Install dir.
                    def installDir = "${INSTALL_DIR_PREFIX}/${RELEASE_NAME}";
                    env.INSTALL_DIR = installDir;
                    env.ARRUS_APPEND_VERSION_SUFFIX_DATE = params.RELEASE ? "OFF" : "ON";

                    // Determine the path where the us4r-api is located.
                    // TODO(US4R-594) this should be removed after splitting ARRUS and HAL.
                    env.US4R_API_RELEASE_DIR = getUs4rApiReleaseDirV2(env);
                }
                sh "pydevops --clean --stage cfg " +
                    "--host '${env.BUILD_ENV_ADDRESS}'  " +
                    "${getDockerOptionsForTemplate(env.DOCKER_OPTIONSv2)}   " +
                    "--src_dir '${env.WORKSPACE}' --build_dir '${env.WORKSPACE}/build'  " +
                    "${env.DOCKER_DIRS}  " +
                    "${env.SSH_DIRS}  " +
                    "--options  " +
                    "build_type='${env.BUILD_TYPE}'  " +
                    "us4r_api_release_dir='${env.US4R_API_RELEASE_DIR}'  " +
                    "/cfg/conan/conan_home='${env.CONAN_HOME_DIR}'  " +
                    "/cfg/conan/profile='${env.TARGET_WORKSPACE_DIR}/.conan/${env.CONAN_PROFILE_FILE}'  " +
                    "/install/prefix='${env.INSTALL_DIR}'  " +
                    "/package_cpp/release_name='${RELEASE_NAME}'  " +
                    "/package_cpp/src_artifact='${env.INSTALL_DIR}/VERSION.rst;${env.INSTALL_DIR}/LICENSE;${env.INSTALL_DIR}/THIRD_PARTY_LICENSES;${env.INSTALL_DIR}/lib64;${env.INSTALL_DIR}/include;${env.INSTALL_DIR}/docs/arrus-cpp.pdf;${env.INSTALL_DIR}/examples'  " +
                    "/package_cpp/dst_dir='${env.TARGET_PRERELEASE_DIR}'   " +
                    "/package_cpp/dst_artifact='${env.CPP_PACKAGE_NAME}'  " +
                    "/package_matlab/release_name='${RELEASE_NAME}'  " +
                    "/package_matlab/src_artifact='${env.INSTALL_DIR}/matlab;${env.INSTALL_DIR}/VERSION.rst'  " +
                    "/package_matlab/dst_dir='${env.TARGET_PRERELEASE_DIR}'   " +
                    "/package_matlab/dst_artifact='${env.MATLAB_PACKAGE_NAME}'  " +
                    "/publish_cpp/release_name='${env.RELEASE_NAME}'  " +
                    "/publish_cpp/target_commitish='${env.BRANCH_NAME}'  " +
                    "/publish_cpp/src_artifact='${env.GITHUB_SOURCE_ARTIFACT_PATH}/${env.CPP_PACKAGE_NAME}*'  " +
                    "/publish_cpp/dst_artifact='__same__'  " +
                    "/publish_cpp/repository_name='pjarosik/arrus'  " +
                    "/publish_cpp/description='${getBuildName(currentBuild)} (C++)'  " +
                    "/publish_matlab/release_name='${env.RELEASE_NAME}'  " +
                    "/publish_matlab/target_commitish='${env.BRANCH_NAME}'  " +
                    "/publish_matlab/src_artifact='${env.GITHUB_SOURCE_ARTIFACT_PATH}/${env.MATLAB_PACKAGE_NAME}*'  " +
                    "/publish_matlab/dst_artifact='__same__'  " +
                    "/publish_matlab/repository_name='pjarosik/arrus'  " +
                    "/publish_matlab/description='${getBuildName(currentBuild)} (MATLAB)'  " +
                    "/publish_py/release_name='${env.RELEASE_NAME}'  " +
                    "/publish_py/target_commitish='${env.BRANCH_NAME}'  " +
                    "/publish_py/src_artifact='${env.GITHUB_PY_ARTIFACT_PATH}/${getArrusWhlNamePattern(params, env.RELEASE_NAME)}'  " +
                    "/publish_py/dst_artifact='__same__'  " +
                    "/publish_py/repository_name='pjarosik/arrus'  " +
                    "/publish_py/description='${getBuildName(currentBuild)} (Python)'  " +
                    "/publish_docs/version='${env.RELEASE_NAME}'  " +
                    "/publish_docs/install_dir='${env.INSTALL_DIR}/'  " +
                    "/cfg/cmake/DARRUS_APPEND_VERSION_SUFFIX_DATE=${env.ARRUS_APPEND_VERSION_SUFFIX_DATE}  " +
                    "/cfg/DARRUS_PY_VERSION=${params.PY_VERSION}  " +
                    "${getPythonExecutableParameter(env, params.PY_VERSION)}  " +
                    "py=ON matlab=ON docs=ON /cfg/cmake/DMatlab_ROOT_DIR=/opt/MATLAB/current"
            }
        }
        stage('Build') {
            when {
                expression { return params.PUBLISH == false }
            }
            steps {
                sh """pydevops --stage build \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${env.DOCKER_DIRS} \
                      ${env.SSH_DIRS}
                   """
            }
        }
        stage('Test') {
            when {
                expression { return params.PUBLISH == false }
            }
            steps {
                sh """pydevops --stage test \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${env.DOCKER_DIRS} \
                      ${env.SSH_DIRS}
                   """
            }
        }
        stage('Install') {
            when {
                expression { return params.PUBLISH == false }
            }
            steps {
                sh """pydevops --stage install \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${DOCKER_DIRS} \
                      ${SSH_DIRS}
                   """
            }
        }
        stage('PackageCpp') {
            when {
                expression { return params.PUBLISH == false }
            }
            steps {
                sh """pydevops --stage package_cpp \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${DOCKER_DIRS} \
                      ${SSH_DIRS}
                   """
            }
        }
        stage('PackageMatlab') {
             when {
                expression { return params.PUBLISH == false }
             }
             steps {
                 sh """pydevops --stage package_matlab \
                       --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                       ${DOCKER_DIRS} \
                       ${SSH_DIRS}
                    """
             }
        }

        // ------------------------------------------ PUBLISH STEPS.
        stage('ValidateRelease') {
            when{
                allOf {
                    expression { params.PUBLISH }
                    expression { params.RELEASE }
                }
            }
            steps {
                script {
                    // C++/MATLAB => unzip the packages and check if the VERRSION.rst has the correct git commit
                    env.CPP_PACKAGE_NAME = us4us.getPackageNameV2(env, params, "${env.JOB_NAME}", "cpp");
                    env.MATLAB_PACKAGE_NAME = us4us.getPackageNameV2(env, params, "${env.JOB_NAME}", "matlab");
                    def packageNames = [env.CPP_PACKAGE_NAME, env.MATLAB_PACKAGE_NAME];

                    packageNames.each { packageName ->
                        def packagePath = "${env.TARGET_PRERELEASE_DIR}/${packageName}.zip"
                        def releasedPackage = "${env.TARGET_RELEASE_DIR}/${packageName}.zip"
                        // Make sure that the version have not been already released.
                        if(us4us.isFileOrDirExists(releasedPackage)) {
                            error "The version ${env.VERSION} has been already released! (remove ${releasedPackage} and Github releases in case you would like to re-release this version)."
                        }
                        // Make sure that the package we publish was generated for the same commit as the current HEAD.
                        // NOTE! It is still possible, that someone will commit something on that branch in between
                        // the ValidateCommit and Publish to repository. However, it seems to be quite unlikely
                        // and can be neglected.
                        def versionFilePath = us4us.extractFileToTempDirectory(packagePath, "VERSION.rst")
                        us4us.validateCommit(versionFilePath)
                    }
                    // Python and docs => check if the Version.rst in the INSTALL_DIR is correct
                    def releaseName = us4us.getReleaseName(env, params);
                    def installDir = "${INSTALL_DIR_PREFIX}/${releaseName}";
                    us4us.validateCommit("${installDir}/VERSION.rst");
                }
            }
        }

        stage('PublishNAS') {
            when {
                allOf {
                    expression { params.PUBLISH }
                    expression { params.RELEASE } // We don't need -dev packages in the `release` directory on NAS.
                }
            }
            steps {
                // copy the package from the pre-release directory to the release directory
                // also, copy the docs directory
                script {
                    def targetFolder = "${env.TARGET_RELEASE_DIR}";
                    sh "mkdir -p ${targetFolder}";
                    sh "mkdir -p ${targetFolder}/docs";
                    // C++/MATLAB => copy the .zip files to the release directory
                    env.CPP_PACKAGE_NAME = us4us.getPackageNameV2(env, params, "${env.JOB_NAME}", "cpp");
                    env.MATLAB_PACKAGE_NAME = us4us.getPackageNameV2(env, params, "${env.JOB_NAME}", "matlab");
                    def packageNames = [env.CPP_PACKAGE_NAME, env.MATLAB_PACKAGE_NAME];

                    packageNames.each { packageName ->
                        def sourceArtifacts = "${TARGET_PRERELEASE_DIR}/${packageName}*"
                        sh "cp ${sourceArtifacts} ${targetFolder}"
                        echo "The files ${sourceArtifacts} were copied to ${targetFolder}"
                    };

                    // TODO consider handling .whl in some other way...
                    // Python and docs => copy the .whl files to the install directory.
                    def releaseName = us4us.getReleaseName(env, params);
                    def installDir = "${INSTALL_DIR_PREFIX}/${releaseName}";
                    sh "cp ${installDir}/python/${getArrusWhlNamePattern(params)} ${targetFolder}";
                    sh "cp -r ${installDir}/docs ${targetFolder}/docs/${releaseName}";
                }
            }
        }
        stage('PublishCpp') {
            when {
                allOf {
                    expression { params.PUBLISH }
                    expression { params.PUBLISH_CPP }
                }
            }
            steps {
                  withCredentials([string(credentialsId: 'us4us-dev-github-token', variable: 'token')]){
                  sh """pydevops --stage publish_cpp \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${DOCKER_DIRS} \
                      ${SSH_DIRS} \
                      --options \
                      token='$token'

                     """
                }
            }
        }
        stage('PublishPython') {
            when {
                allOf {
                    expression { params.PUBLISH }
                    expression { params.PUBLISH_PY }
                }
            }
            steps {
                  withCredentials([string(credentialsId: 'us4us-dev-github-token', variable: 'token')]){
                  sh """pydevops --stage publish_py \
                     --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                     ${DOCKER_DIRS} \
                     ${SSH_DIRS} \
                     --options \
                     token='$token' \
                     """
                }
            }
        }
        stage('PublishMatlab') {
            when {
                allOf {
                    expression { params.PUBLISH }
                    expression { params.PUBLISH_MATLAB }
                }
            }
            steps {
                  withCredentials([string(credentialsId: 'us4us-dev-github-token', variable: 'token')]){
                  sh """pydevops --stage publish_matlab \
                     --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                     ${DOCKER_DIRS} \
                     ${SSH_DIRS} \
                     --options \
                     token='$token'
                     """
                }
            }
        }
        stage('PublishDocs') {
            when {
                allOf {
                    expression { params.PUBLISH }
                    expression { params.PUBLISH_DOCS }
                }
             }
             steps {
                   withCredentials([usernamePassword(credentialsId: 'us4us-dev-github-credentials', usernameVariable: 'username', passwordVariable: 'password')]){
                   sh """pydevops --stage publish_docs \
                      --src_dir='${env.WORKSPACE}' --build_dir='${env.WORKSPACE}/build' \
                      ${DOCKER_DIRS} \
                      ${SSH_DIRS} \
                      --options \
                      repository='https://$username:$password@github.com/us4useu/arrus-docs.git' \
                      commit_msg='Updated docs, ${getBuildName(currentBuild)}'
                      """
                 }
             }
         }
    }
     post {
         failure {
             script {
                 emailext(body: "Check console output at $BUILD_URL to view the results.",
                    from: 'us4usdevs@gmail.com', replyTo: 'dev@us4us.eu',
                    recipientProviders: [developers(), requestor()],
                    subject: "Build failed in Jenkins: $JOB_NAME")
             }
         }
         unstable {
             script {
                 emailext(body: "Check console output at $BUILD_URL to view the results.",
                    from: 'us4usdevs@gmail.com', replyTo: 'dev@us4us.eu',
                    recipientProviders: [developers(), requestor()],
                    subject: "Unstable build in Jenkins: $JOB_NAME")
             }
         }
         changed {
             script {
                emailext(body:    "Check console output at $BUILD_URL to view the results.",
                    from: 'us4usdevs@gmail.com', replyTo: 'dev@us4us.eu',
                    recipientProviders: [developers(), requestor()],
                    subject: "Jenkins build is back to normal: $JOB_NAME")
             }
         }
     }
}

def getArrusWhlNamePattern(params, releaseName) {
    pythonVersion = "cp${params.PY_VERSION}".replace(".", "");
    // releaseName can be e.g. v0.12.0-dev, but whl will be always v0.12.0.dev
    whlReleaseName = releaseName.replace("-dev", ".dev");
    if(us4us.isPrereleaseV2(params)) {
        return "arrus*${whlReleaseName}*${us4us.getTimestamp()}*${pythonVersion}*.whl";
    }
    else {
        return "arrus*${whlReleaseName}*${pythonVersion}*.whl";
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
    def sanitizedPythonVersion = pythonVersion.replace(".", "");
    def pythonExecutablePath = us4us.getUs4usJenkinsVariable(env, "ARRUS_PYTHON_EXECUTABLE_${sanitizedPythonVersion}");
    if(pythonExecutablePath != null && !pythonExecutablePath.trim().isEmpty()) {
        return "/cfg/DPYTHON_EXECUTABLE=${pythonExecutablePath}";
    }
    else {
        return "";
    }
}

def isSCMOnly(params) {
    // note: the fact that env.SCM_ONLY is null on the first call seems to be a bug 
    // . Currently this is a way to detect if this is the first build of the new branch
    // however in the future releases of Jenkins this may change.
    return (params.SCM_ONLY == null || params.SCM_ONLY == 'true')
}

/**
 Returns the path to the unzipped us4r-api package.
 Currently, it is basically the path to "unzipped" directory in the pre-release directory.
 */
def getUs4rApiReleaseDirV2(env) {
    def nasDir = us4us.getUs4usJenkinsVariable(env, "NAS_DIR");
    def platformName = us4us.getPlatformNameAndBuildType("${env.JOB_NAME}");
    return "${nasDir}/us4r-hal/pre-release/${platformName}/unzipped/";
}
