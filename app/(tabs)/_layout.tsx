import { Tabs } from 'expo-router';
import { SymbolView, type SymbolViewProps } from 'expo-symbols';
import { StyleSheet, Text, View, useWindowDimensions, type ColorValue } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useReduceMotionEnabled } from '@/accessibility/use-accessibility-preferences';
import { colors, fonts, radii } from '@/theme';

const TAB_BAR_CONTENT_HEIGHT = 64;

type TabIconProps = {
  color: ColorValue;
  focused: boolean;
  name: SymbolViewProps['name'];
  size: number;
};

function TabIcon({ color, focused, name, size }: TabIconProps) {
  const { fontScale } = useWindowDimensions();
  // Expo Symbols uses a font glyph on Android; give its scaled glyph room.
  const iconSize = size * Math.max(1, fontScale);
  return (
    <View
      accessible={false}
      style={[styles.iconContainer, { minHeight: iconSize, minWidth: Math.max(56, iconSize) }, focused && styles.iconContainerSelected]}
    >
      <SymbolView
        accessible={false}
        name={name}
        size={size}
        style={{ width: iconSize, height: iconSize }}
        tintColor={color}
      />
    </View>
  );
}

export default function PrimaryTabsLayout() {
  const reduceMotionEnabled = useReduceMotionEnabled();
  const insets = useSafeAreaInsets();
  const { fontScale, width } = useWindowDimensions();
  const labelWidth = Math.max(56, width / 4 - 16);
  const labelLines = Math.max(1, Math.ceil((72 * fontScale) / labelWidth));
  const iconHeight = Math.max(28, 25 * fontScale);
  const tabHeight = Math.max(TAB_BAR_CONTENT_HEIGHT, iconHeight + 12 + 16 * fontScale * labelLines);

  return (
    <View
      style={[styles.container, { paddingTop: insets.top }]}
      testID="primary-tabs-safe-area"
    >
      <Tabs
        screenOptions={{
          animation: reduceMotionEnabled ? 'none' : 'shift',
          headerShown: false,
          lazy: true,
          sceneStyle: { backgroundColor: colors.canvas },
          tabBarActiveTintColor: colors.primary,
          tabBarAllowFontScaling: true,
          tabBarHideOnKeyboard: true,
          tabBarInactiveTintColor: colors.text,
          tabBarItemStyle: styles.tabItem,
          tabBarIconStyle: { height: iconHeight, width: Math.max(56, iconHeight) },
          tabBarLabel: ({ children, color }) => (
            <Text accessible={false} style={[styles.tabLabel, { color, maxWidth: labelWidth }]}>
              {children}
            </Text>
          ),
          tabBarLabelPosition: 'below-icon',
          tabBarLabelStyle: styles.tabLabel,
          tabBarStyle: [
            styles.tabBar,
            { height: tabHeight + insets.bottom },
          ],
        }}
      >
        <Tabs.Screen
          name="index"
          options={{
            tabBarAccessibilityLabel: 'Beranda',
            tabBarButtonTestID: 'tab-home',
            tabBarIcon: ({ color, focused, size }) => (
              <TabIcon
                color={color}
                focused={focused}
                name={{ android: 'home', ios: 'house' }}
                size={size}
              />
            ),
            title: 'Beranda',
          }}
        />
        <Tabs.Screen
          name="live"
          options={{
            tabBarAccessibilityLabel: 'Pindai',
            tabBarButtonTestID: 'tab-scan',
            tabBarIcon: ({ color, focused, size }) => (
              <TabIcon
                color={color}
                focused={focused}
                name={{ android: 'center_focus_strong', ios: 'viewfinder' }}
                size={size}
              />
            ),
            title: 'Pindai',
          }}
        />
        <Tabs.Screen
          name="translator"
          options={{
            tabBarAccessibilityLabel: 'Penerjemah',
            tabBarButtonTestID: 'tab-translator',
            tabBarIcon: ({ color, focused, size }) => (
              <TabIcon
                color={color}
                focused={focused}
                name={{ android: 'translate', ios: 'character.bubble' }}
                size={size}
              />
            ),
            title: 'Penerjemah',
          }}
        />
        <Tabs.Screen
          name="settings"
          options={{
            tabBarAccessibilityLabel: 'Setelan',
            tabBarButtonTestID: 'tab-settings',
            tabBarIcon: ({ color, focused, size }) => (
              <TabIcon
                color={color}
                focused={focused}
                name={{ android: 'settings', ios: 'gearshape' }}
                size={size}
              />
            ),
            title: 'Setelan',
          }}
        />
      </Tabs>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.canvas,
  },
  tabBar: {
    backgroundColor: colors.surfaceContainerLowest,
    borderTopColor: colors.outlineVariant,
    borderTopWidth: 1,
    elevation: 8,
  },
  tabItem: {
    minHeight: 56,
    borderCurve: 'continuous',
    borderRadius: radii.md,
    marginHorizontal: 4,
    marginVertical: 4,
  },
  tabLabel: {
    fontFamily: fonts.bodySemibold,
    fontSize: 12,
    fontWeight: '600',
    lineHeight: 16,
    textAlign: 'center',
    flexShrink: 1,
  },
  iconContainer: {
    minHeight: 28,
    minWidth: 56,
    alignItems: 'center',
    justifyContent: 'center',
    borderCurve: 'continuous',
    borderRadius: radii.pill,
  },
  iconContainerSelected: {
    backgroundColor: colors.primaryContainer,
  },
});
