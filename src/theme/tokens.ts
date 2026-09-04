export const colors = {
  background: '#F5F7FA',
  surface: '#F5F7FA',
  surfaceDim: '#D6DEE8',
  surfaceBright: '#FFFFFF',
  surfaceContainerLowest: '#FFFFFF',
  surfaceContainerLow: '#EEF3F7',
  surfaceContainer: '#E6EDF3',
  surfaceContainerHigh: '#DEE7EE',
  surfaceContainerHighest: '#D6DEE8',
  text: '#102A43',
  textSecondary: '#486581',
  inverseSurface: '#102A43',
  inverseOnSurface: '#FFFFFF',
  outline: '#526D82',
  outlineVariant: '#BCCCDC',
  primary: '#006B63',
  onPrimary: '#FFFFFF',
  primaryContainer: '#D5F3EF',
  onPrimaryContainer: '#0A4540',
  secondary: '#725B00',
  onSecondary: '#FFFFFF',
  secondaryContainer: '#FFF1B8',
  onSecondaryContainer: '#493B00',
  error: '#B42318',
  onError: '#FFFFFF',
  errorContainer: '#FEE4E2',
  onErrorContainer: '#7A271A',
  success: '#087443',
  successContainer: '#D1FADF',
  warning: '#725B00',
  warningContainer: '#FFF1B8',

  // Compatibility aliases used by existing feature styles.
  canvas: '#F5F7FA',
  accent: '#006B63',
  border: '#BCCCDC',
} as const;

export const fonts = {
  heading: 'HankenGrotesk_600SemiBold',
  headingBold: 'HankenGrotesk_700Bold',
  body: 'Inter_400Regular',
  bodyMedium: 'Inter_500Medium',
  bodySemibold: 'Inter_600SemiBold',
  monoMedium: 'JetBrainsMono_500Medium',
} as const;

export const spacing = {
  xxs: 4,
  xs: 8,
  sm: 12,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 40,
  section: 64,
  sectionDesktop: 120,
} as const;

export const radii = {
  sm: 8,
  default: 12,
  md: 16,
  lg: 20,
  xl: 24,
  pill: 9999,
} as const;

export const typography = {
  displayLarge: {
    fontFamily: fonts.headingBold,
    fontSize: 48,
    fontWeight: '700',
    letterSpacing: -0.96,
    lineHeight: 56,
  },
  displayLargeMobile: {
    fontFamily: fonts.headingBold,
    fontSize: 32,
    fontWeight: '700',
    letterSpacing: -0.32,
    lineHeight: 40,
  },
  headlineMedium: {
    fontFamily: fonts.heading,
    fontSize: 24,
    fontWeight: '600',
    lineHeight: 32,
  },
  bodyLarge: {
    fontFamily: fonts.body,
    fontSize: 18,
    fontWeight: '400',
    lineHeight: 28,
  },
  body: {
    fontFamily: fonts.body,
    fontSize: 16,
    fontWeight: '400',
    lineHeight: 24,
  },
  label: {
    fontFamily: fonts.bodySemibold,
    fontSize: 16,
    fontWeight: '600',
    lineHeight: 20,
  },
  heading: {
    fontFamily: fonts.heading,
    fontSize: 24,
    fontWeight: '600',
    lineHeight: 32,
  },
  title: {
    fontFamily: fonts.headingBold,
    fontSize: 32,
    fontWeight: '700',
    letterSpacing: -0.32,
    lineHeight: 40,
  },
  supporting: {
    fontFamily: fonts.body,
    fontSize: 14,
    fontWeight: '400',
    lineHeight: 20,
  },
  labelCaps: {
    fontFamily: fonts.monoMedium,
    fontSize: 12,
    fontWeight: '500',
    letterSpacing: 0.6,
    lineHeight: 16,
  },
} as const;

export const minimumTouchTarget = 48;
