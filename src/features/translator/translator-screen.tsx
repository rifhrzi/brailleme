import { memo, useMemo, useSyncExternalStore } from 'react';
import {
  KeyboardAvoidingView,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  useWindowDimensions,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import { colors, radii, spacing, typography } from '@/theme';

import {
  decodeCells,
  encodeText,
  MAX_BRAILLE_CELLS,
  MAX_TEXT_CHARACTERS,
  type BrailleItem,
  type TranslationIssue,
} from './codec';
import {
  getTranslatorDraft,
  subscribeTranslatorDraft,
  updateTranslatorDraft,
  type TranslatorMode,
} from './draft';
import { useTranslatorKeyboard } from './use-translator-keyboard';

const DOT_ROWS = [[1, 4], [2, 5], [3, 6]] as const;
const MODES: { value: TranslatorMode; label: string }[] = [
  { value: 'text-to-braille', label: 'Teks ke Braille' },
  { value: 'braille-to-text', label: 'Braille ke Teks' },
];

function itemDescription(item: BrailleItem, index: number) {
  if (item.kind === 'space') return `Sel ${index + 1}, spasi`;
  if (item.kind === 'newline') return `Sel ${index + 1}, baris baru`;
  const dots = [1, 2, 3, 4, 5, 6].filter((dot) => item.mask & (1 << (dot - 1)));
  return `Sel ${index + 1}, ${dots.length ? `titik ${dots.join(', ')}` : 'kosong, belum lengkap'}`;
}

const CellDots = memo(function CellDots({ mask }: { mask: number }) {
  return (
    <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={styles.cellDots}>
      {DOT_ROWS.map((row, index) => (
        <View key={index} style={styles.previewDotRow}>
          {row.map((dot) => (
            <View key={dot} style={[styles.previewDot, mask & (1 << (dot - 1)) ? styles.dotActive : styles.dotInactive]} />
          ))}
        </View>
      ))}
    </View>
  );
});

const PreviewCell = memo(function PreviewCell({ kind, mask, index, editable, selected }: {
  kind: BrailleItem['kind'];
  mask: number;
  index: number;
  editable: boolean;
  selected: boolean;
}) {
  const label = itemDescription(kind === 'cell' ? { kind, mask } : { kind }, index);
  const content = <>
    {kind === 'cell' ? <CellDots mask={mask} /> : <Text style={styles.itemMarker}>{kind === 'space' ? 'Spasi' : 'Baris baru'}</Text>}
    <Text style={styles.cellNumber}>{index + 1}</Text>
  </>;
  return editable ? (
    <Pressable
      accessibilityLabel={label}
      accessibilityRole="button"
      accessibilityState={{ selected }}
      onPress={() => updateTranslatorDraft({ type: 'select', index })}
      style={({ pressed }) => [styles.previewCell, selected && styles.cellSelected, pressed && styles.pressed]}
      testID={`braille-cell-${index}`}
    >{content}</Pressable>
  ) : (
    <View accessible accessibilityLabel={label} style={styles.previewCell} testID={`braille-cell-${index}`}>{content}</View>
  );
});

function Issues({ issues, mode }: { issues: TranslationIssue[]; mode: TranslatorMode }) {
  if (!issues.length) return null;
  return (
    <View accessibilityLiveRegion="polite" style={styles.error} testID="translation-issues">
      <Text accessibilityRole="alert" selectable style={styles.errorTitle}>Periksa masukan</Text>
      {issues.slice(0, 3).map((issue, index) => (
        <Text key={`${issue.index}-${issue.code}-${index}`} selectable style={styles.errorCopy}>
          {mode === 'text-to-braille' ? 'Karakter' : 'Sel'} {issue.index + 1}: {issue.message}
        </Text>
      ))}
      {issues.length > 3 ? <Text selectable style={styles.errorCopy}>Masih ada {issues.length - 3} bagian yang perlu diperiksa.</Text> : null}
    </View>
  );
}

export function TranslatorScreen() {
  const state = useSyncExternalStore(subscribeTranslatorDraft, getTranslatorDraft, getTranslatorDraft);
  const { width, fontScale } = useWindowDimensions();
  const insets = useSafeAreaInsets();
  const { scrollRef, inputRef, keepInputVisible, onFocus, onBlur } = useTranslatorKeyboard(insets.top);
  const compact = width < 380 || fontScale >= 1.5;
  const textMode = state.mode === 'text-to-braille';
  const encoded = useMemo(() => encodeText(state.text), [state.text]);
  const decoded = useMemo(() => decodeCells(state.items), [state.items]);
  const result = textMode ? encoded : decoded;
  const preview = textMode ? (encoded.valid ? encoded.value : []) : state.items;
  const selected = state.selectedIndex === null ? undefined : state.items[state.selectedIndex];
  const textCount = [...state.text].length;
  const rows: { item: BrailleItem; index: number }[][] = [[]];
  preview.forEach((item, index) => {
    rows[rows.length - 1]?.push({ item, index });
    if (item.kind === 'newline') rows.push([]);
  });

  return (
    <KeyboardAvoidingView
      behavior={process.env.EXPO_OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={insets.top}
      style={styles.container}
      testID="translator-keyboard-viewport"
    >
      <Screen
        automaticallyAdjustKeyboardInsets
        keyboardDismissMode="on-drag"
        onContentSizeChange={keepInputVisible}
        onLayout={keepInputVisible}
        scrollRef={scrollRef}
        testID="translator-screen"
      >
        <Text accessibilityRole="header" selectable style={styles.heading}>Penerjemah</Text>
        <Text selectable style={styles.intro}>Ketik teks atau susun titik Braille. Hasil berubah saat Anda mengedit.</Text>
        <View accessibilityRole="tablist" style={[styles.modeTabs, compact && styles.stacked]} testID="translator-mode-switch">
          {MODES.map((mode) => (
            <Pressable
              key={mode.value}
              accessibilityLabel={mode.label}
              accessibilityRole="tab"
              accessibilityState={{ selected: state.mode === mode.value }}
              onPress={() => updateTranslatorDraft({ type: 'mode', mode: mode.value })}
              style={({ pressed }) => [styles.modeTab, state.mode === mode.value && styles.modeSelected, pressed && styles.pressed]}
            >
              <Text style={[styles.modeLabel, state.mode === mode.value && styles.modeLabelSelected]}>{mode.label}</Text>
            </Pressable>
          ))}
        </View>
        {state.notice ? <Text accessibilityLiveRegion="polite" selectable style={styles.notice}>{state.notice}</Text> : null}

        <SectionCard title="Braille" style={styles.card}>
          <Text selectable style={styles.supporting}>
            {textMode ? 'Tanda kapital dan angka ikut ditampilkan.' : 'Pilih sel untuk mengubah titiknya. Sel baru disisipkan setelah pilihan Anda.'}
          </Text>
          {preview.length ? (
            <ScrollView nestedScrollEnabled style={styles.previewVertical} testID="braille-preview">
              <ScrollView horizontal nestedScrollEnabled contentContainerStyle={styles.previewContent}>
                <View style={styles.previewLines}>
                  {rows.map((row, rowIndex) => (
                    <View key={rowIndex} style={styles.previewRow}>
                      {row.map(({ item, index }) => <PreviewCell
                        editable={!textMode}
                        index={index}
                        key={index}
                        kind={item.kind}
                        mask={item.kind === 'cell' ? item.mask : 0}
                        selected={!textMode && state.selectedIndex === index}
                      />)}
                    </View>
                  ))}
                </View>
              </ScrollView>
            </ScrollView>
          ) : <Text selectable style={styles.empty}>{textMode ? (encoded.valid ? 'Braille akan muncul setelah Anda mengetik.' : 'Perbaiki teks untuk melihat hasil Braille.') : 'Pilih Tambah sel, lalu aktifkan titiknya.'}</Text>}
          <Text selectable style={styles.counter} testID="braille-count">
            {textMode ? encoded.value.length : state.items.length} / {MAX_BRAILLE_CELLS} sel, termasuk spasi dan baris baru
          </Text>
        </SectionCard>

        <SectionCard title={textMode ? 'Teks' : 'Hasil teks'} style={styles.card}>
          {textMode ? <>
            <TextInput
              accessibilityLabel="Teks yang diterjemahkan"
              accessibilityHint="Ketik huruf, angka, spasi, baris baru, atau tanda baca. Maksimal 200 karakter."
              allowFontScaling
              autoCapitalize="none"
              autoCorrect={false}
              multiline
              onBlur={onBlur}
              onChangeText={(text) => updateTranslatorDraft({ type: 'text', text })}
              onContentSizeChange={keepInputVisible}
              onFocus={onFocus}
              placeholder="Contoh: Baca? 001"
              placeholderTextColor={colors.textSecondary}
              ref={inputRef}
              style={styles.textInput}
              testID="translator-text-input"
              value={state.text}
            />
            <Text selectable style={styles.counter}>{textCount} / {MAX_TEXT_CHARACTERS} karakter</Text>
          </> : <>
            <Text accessibilityLiveRegion="polite" selectable style={styles.output} testID="translated-text">
              {decoded.valid ? (decoded.value || 'Teks akan muncul setelah Anda menyusun Braille.') : 'Perbaiki susunan Braille untuk melihat hasil teks.'}
            </Text>
            <Text selectable style={styles.counter}>{[...decoded.value].length} / {MAX_TEXT_CHARACTERS} karakter</Text>
          </>}
          <Issues issues={result.issues} mode={state.mode} />
          {textMode ? <AccessibleButton disabled={!state.text.length} label="Hapus semua" onPress={() => updateTranslatorDraft({ type: 'clear' })} style={styles.touchTarget} variant="secondary" /> : null}
        </SectionCard>

        {!textMode ? <SectionCard title="Edit titik" style={styles.card}>
          <Text accessibilityLiveRegion="polite" selectable style={styles.body}>
            {selected?.kind === 'cell' ? `Sel ${state.selectedIndex! + 1} dipilih. Titik kiri 1, 2, 3; titik kanan 4, 5, 6.` : selected ? `${selected.kind === 'space' ? 'Spasi' : 'Baris baru'} dipilih. Pilih sebuah sel untuk mengedit titik.` : 'Tambahkan sel untuk mulai menyusun titik.'}
          </Text>
          <View style={styles.dotControls}>
            {DOT_ROWS.map((row, index) => <View key={index} style={styles.dotControlRow}>
              {row.map((dot) => {
                const checked = selected?.kind === 'cell' && Boolean(selected.mask & (1 << (dot - 1)));
                const disabled = selected?.kind !== 'cell';
                return <Pressable
                  accessibilityLabel={`Titik ${dot}, ${checked ? 'aktif' : 'tidak aktif'}`}
                  accessibilityRole="checkbox"
                  accessibilityState={{ checked, disabled }}
                  disabled={disabled}
                  key={dot}
                  onPress={() => updateTranslatorDraft({ type: 'toggle-dot', dot })}
                  style={({ pressed }) => [styles.dotControl, checked && styles.dotControlSelected, disabled && styles.disabled, pressed && styles.pressed]}
                  testID={`translator-dot-${dot}`}
                >
                  <Text style={styles.dotNumber}>{dot}</Text>
                  <Text style={styles.dotState}>{checked ? 'Aktif' : 'Tidak aktif'}</Text>
                </Pressable>;
              })}
            </View>)}
          </View>
          <Text selectable style={styles.supporting}>Sel tanpa titik belum lengkap. Gunakan Spasi untuk memisahkan kata.</Text>
          <View style={styles.actions}>
            <AccessibleButton label="Tambah sel" onPress={() => updateTranslatorDraft({ type: 'insert', item: { kind: 'cell', mask: 0 } })} style={styles.touchTarget} variant="action" />
            <AccessibleButton label="Spasi" onPress={() => updateTranslatorDraft({ type: 'insert', item: { kind: 'space' } })} style={styles.touchTarget} variant="secondary" />
            <AccessibleButton label="Baris baru" onPress={() => updateTranslatorDraft({ type: 'insert', item: { kind: 'newline' } })} style={styles.touchTarget} variant="secondary" />
            <AccessibleButton disabled={state.selectedIndex === null} label="Hapus sel" onPress={() => updateTranslatorDraft({ type: 'delete' })} style={styles.touchTarget} variant="secondary" />
            <AccessibleButton disabled={!state.items.length} label="Hapus semua" onPress={() => updateTranslatorDraft({ type: 'clear' })} style={styles.touchTarget} variant="secondary" />
          </View>
        </SectionCard> : null}
        <Text selectable style={styles.supporting}>Gunakan huruf A–Z, angka bulat, serta tanda . , ? ! : ; - ( ). Pisahkan huruf dan angka dengan spasi. Draf hanya tersedia selama aplikasi terbuka.</Text>
      </Screen>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.canvas },
  heading: { ...typography.title, color: colors.text },
  intro: { ...typography.body, color: colors.textSecondary },
  body: { ...typography.body, color: colors.text },
  supporting: { ...typography.supporting, color: colors.textSecondary },
  card: { padding: spacing.md },
  modeTabs: { flexDirection: 'row', gap: spacing.xs },
  stacked: { flexDirection: 'column' },
  modeTab: { flex: 1, minHeight: 56, justifyContent: 'center', borderRadius: radii.default, borderWidth: 2, borderColor: colors.outline, padding: spacing.md },
  modeSelected: { backgroundColor: colors.primaryContainer, borderColor: colors.primary, borderBottomWidth: 4 },
  modeLabel: { ...typography.label, color: colors.text, textAlign: 'center', flexShrink: 1 },
  modeLabelSelected: { color: colors.onPrimaryContainer },
  pressed: { opacity: 0.75 },
  notice: { ...typography.body, color: colors.onSecondaryContainer, backgroundColor: colors.secondaryContainer, padding: spacing.md, borderRadius: radii.default },
  previewVertical: { maxHeight: 280 },
  previewContent: { paddingVertical: spacing.xxs },
  previewLines: { gap: spacing.xs },
  previewRow: { flexDirection: 'row', alignItems: 'stretch', gap: spacing.xs, minHeight: 56 },
  previewCell: { minWidth: 56, minHeight: 88, borderWidth: 2, borderColor: colors.outlineVariant, borderRadius: radii.sm, padding: spacing.xs, gap: spacing.xs, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.surfaceContainerLowest },
  cellSelected: { borderColor: colors.primary, borderWidth: 3, backgroundColor: colors.primaryContainer },
  cellDots: { gap: 5 },
  previewDotRow: { flexDirection: 'row', gap: 8 },
  previewDot: { width: 10, height: 10, borderRadius: 5, borderWidth: 1, borderColor: colors.textSecondary },
  dotActive: { backgroundColor: colors.text, borderColor: colors.text },
  dotInactive: { backgroundColor: colors.surfaceContainerLowest },
  cellNumber: { ...typography.supporting, color: colors.textSecondary },
  itemMarker: { ...typography.supporting, color: colors.text, textAlign: 'center' },
  empty: { ...typography.body, color: colors.textSecondary, paddingVertical: spacing.md },
  counter: { ...typography.supporting, color: colors.textSecondary, fontVariant: ['tabular-nums'] },
  textInput: { ...typography.bodyLarge, minHeight: 160, color: colors.text, backgroundColor: colors.surfaceContainerLowest, borderWidth: 2, borderColor: colors.outline, borderRadius: radii.default, padding: spacing.md, textAlignVertical: 'top' },
  output: { ...typography.bodyLarge, color: colors.text, minHeight: 56 },
  error: { backgroundColor: colors.errorContainer, padding: spacing.md, borderRadius: radii.default, gap: spacing.xs },
  errorTitle: { ...typography.label, color: colors.onErrorContainer },
  errorCopy: { ...typography.body, color: colors.onErrorContainer },
  dotControls: { gap: spacing.sm },
  dotControlRow: { flexDirection: 'row', gap: spacing.sm },
  dotControl: { flex: 1, minWidth: 56, minHeight: 80, padding: spacing.sm, borderRadius: radii.default, borderWidth: 2, borderColor: colors.outline, alignItems: 'center', justifyContent: 'center', gap: spacing.xxs },
  dotControlSelected: { backgroundColor: colors.primaryContainer, borderColor: colors.primary },
  dotNumber: { ...typography.heading, color: colors.text },
  dotState: { ...typography.supporting, color: colors.text, textAlign: 'center', flexShrink: 1 },
  disabled: { opacity: 0.5 },
  actions: { gap: spacing.sm },
  touchTarget: { minHeight: 56 },
});
