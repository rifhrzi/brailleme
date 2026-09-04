import { Tabs } from 'expo-router';
import { SymbolView, type SymbolViewProps } from 'expo-symbols';
import { StyleSheet, View, type ColorValue } from 'react-native';
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
  return (
    <View
      accessible={false}
      style={[styles.iconContainer, focused && styles.iconContainerSelected]}
    >
      <SymbolView
        accessible={false}
        name={name}
        size={size}
        tintColor={color}
      />
    </View>
  );
}

export default function PrimaryTabsLayout() {
  const reduceMotionEnabled = useReduceMotionEnabled();
  const insets = useSafeAreaInsets();

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
          tabBarLabelStyle: styles.tabLabel,
          tabBarStyle: [
            styles.tabBar,
            { height: TAB_BAR_CONTENT_HEIGHT + insets.bottom },
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
          name="settings"
          options={{
            tabBarAccessibilityLabel: 'Pengaturan',
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
