import type { NextPage, GetStaticProps } from 'next';
import NavBar from '@/components/NavBar';
import Hero from '@/components/Hero';
import Problem from '@/components/Problem';
import Approach from '@/components/Approach';
import Results from '@/components/Results';
import Demo from '@/components/Demo';
import About from '@/components/About';

interface HomeProps {
  basePath: string;
}

const Home: NextPage<HomeProps> = ({ basePath }) => (
  <>
    <NavBar />
    <Hero />
    <Problem />
    <Approach />
    <Results basePath={basePath} />
    <Demo basePath={basePath} />
    <About />
  </>
);

export const getStaticProps: GetStaticProps<HomeProps> = () => ({
  props: { basePath: process.env.NEXT_PUBLIC_BASE_PATH ?? '' },
});

export default Home;
