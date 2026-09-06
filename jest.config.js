/** @type {import('jest').Config} */
module.exports = {
  preset: 'jest-expo',
  clearMocks: true,
  modulePathIgnorePatterns: [
    '<rootDir>/.braille-training-work/',
    '<rootDir>/training/evidence/',
  ],
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/__tests__/**',
    '!src/**/index.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/types/app-preferences.ts',
    '!src/types/runtime.ts',
    '!src/services/ports.ts',
  ],
  coverageThreshold: {
    global: {
      statements: 70,
      branches: 70,
      functions: 70,
      lines: 70,
    },
  },
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '\\.tflite$': '<rootDir>/scripts/jest-model-asset.cjs',
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  testPathIgnorePatterns: ['/node_modules/', '/dist/', '/.braille-training-work/', '/training/evidence/'],
};
