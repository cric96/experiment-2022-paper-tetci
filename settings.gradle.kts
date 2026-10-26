plugins {
    id("com.gradle.develocity") version "4.6.1"
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

develocity {
    buildScan {
        termsOfUseUrl = "https://gradle.com/help/legal-terms-of-use"
        termsOfUseAgree = "yes"
        publishing.onlyIf { it.buildResult.failures.isNotEmpty() }
    }
}

rootProject.name = "experiment-2023-paper-tetci"
