plugins {
    id("com.gradle.enterprise") version "3.16"
}

gradleEnterprise {
    buildScan {
        termsOfServiceUrl = "https://gradle.com/terms-of-service"
        termsOfServiceAgree = "yes"
        publishOnFailure()
    }
}

rootProject.name = "experiement-2023-paper-tetci"
