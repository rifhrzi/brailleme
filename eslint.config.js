const { defineConfig } = require('eslint/config');
const expoConfig = require('eslint-config-expo/flat');

module.exports = defineConfig([
  ...expoConfig,
  {
    ignores: ['coverage/**', 'dist/**', 'output/**', 'training/evidence/**', 'training/checkpoints/**', '.braille-training-work/**'],
  },
]);
