import { PrimaryPage } from '@/components';
import HomeScreen from '@/features/home/home-screen';

export default function HomeRoute() {
  return (
    <PrimaryPage page="home">
      <HomeScreen />
    </PrimaryPage>
  );
}
