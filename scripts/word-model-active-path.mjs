import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import ts from 'typescript';

// Follow the same literal Metro asset as the app; historical model directories
// remain intact and cannot accidentally satisfy the active release checks.
export function activeWordModelDirectory(root) {
  const syntax = ts.createSourceFile('word-model-assets.ts',
    readFileSync(resolve(root, 'src/ml/word-model-assets.ts'), 'utf8'), ts.ScriptTarget.Latest, true);
  const declarations = syntax.statements.filter((statement) => ts.isVariableStatement(statement) &&
    statement.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword))
    .flatMap((statement) => statement.declarationList.declarations);
  const value = declarations.find((declaration) => ts.isIdentifier(declaration.name) &&
    declaration.name.text === 'WORD_MODEL_ASSET')?.initializer;
  assert.ok(value && ts.isCallExpression(value) && ts.isIdentifier(value.expression) && value.expression.text === 'require' &&
    value.arguments.length === 1 && ts.isStringLiteral(value.arguments[0]), 'WORD_MODEL_ASSET must export the actual literal Metro asset');
  const match = /^\.\.\/\.\.\/assets\/models\/(braille-word-reader-v1|braille-word-reader-id-v5)\/[^/\\]+\.tflite$/.exec(value.arguments[0].text);
  assert.ok(match, 'Active word model must use an explicitly supported model directory');
  return resolve(root, 'assets/models', match[1]);
}

export function activeWordModelManifest(root) {
  return JSON.parse(readFileSync(resolve(activeWordModelDirectory(root), 'manifest.json'), 'utf8'));
}
