const path = require('node:path');
module.exports = {
  rootDir: path.join(__dirname, 'source-private'),
  preset: 'jest-expo',
  moduleNameMapper: { '^@/(.*)$': '<rootDir>/src/$1' },
  testMatch: ['<rootDir>/src/ml/__tests__/word-decoder-positions.test.ts'],
  cacheDirectory: path.join(__dirname, 'jest-cache'),
  collectCoverage: false,
};
