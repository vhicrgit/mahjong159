plugins {
    id("com.android.application")
}

android {
    namespace = "com.mahjong159.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.mahjong159.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
