import Constants from 'expo-constants';

export type AppEnvironment = 'development' | 'preview' | 'production';

const configuredEnvironment = Constants.expoConfig?.extra?.appEnvironment;

export const APP_ENVIRONMENT: AppEnvironment =
  configuredEnvironment === 'preview' || configuredEnvironment === 'production'
    ? configuredEnvironment
    : 'development';

export const IS_PRODUCTION_BUILD = APP_ENVIRONMENT === 'production';

export const ALLOW_UNVERIFIED_MODEL_IN_PRODUCTION =
  IS_PRODUCTION_BUILD &&
  Constants.expoConfig?.extra?.allowUnverifiedModel === true;
