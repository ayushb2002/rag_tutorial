from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

sample_text = """
Black holes are some of the most fascinating and mind-bending objects in the cosmos. The very thing that characterizes a black hole also makes it hard to study: its intense gravity. All the mass in a black hole is concentrated in a tiny region, surrounded by a boundary called the “event horizon”. Nothing that crosses that boundary can return to the outside universe, not even light. A black hole itself is invisible.

But astronomers can still observe black holes indirectly by the way their gravity affects stars and pulls matter into orbit. As gas flows around a black hole, it heats up, paradoxically making these invisible objects into some of the brightest things in the entire universe. As a result, we can see some black holes from billions of light-years away. For one large black hole in a nearby galaxy, astronomers even managed to see a ring of light around the event horizon, using a globe-spanning array of powerful telescopes.

Our Work
Center for Astrophysics | Harvard & Smithsonian scientists participate in many black hole-related projects:

Using the Event Horizon Telescope (EHT) to capture the first image of a black hole's “shadow”: the absence of light that marks where the event horizon is located. The EHT is composed of many telescopes working together to create one Earth-sized observatory, all monitoring the supermassive black hole at the center of the galaxy M87, leading to the first image ever captured of a black hole.
CfA Plays Central Role In Capturing Landmark Black Hole Image

Observing supermassive black holes in other galaxies to understand how they evolve and shape their host galaxies. CfA astronomers use telescopes across the entire spectrum of light, from radio waves to X-rays to gamma rays.
A Surprising Blazar Connection Revealed

Studying the infall of matter — called “accretion” — onto black holes, using NASA's Chandra X-ray Observatory and other telescopes. In addition, CfA researchers use cutting-edge supercomputers to create theoretical models for the disks and jets of matter that black holes create around themselves.
Supermassive Black Hole Spins Super-Fast

Hunting for black hole interactions with other astronomical objects. That includes “disruption” events, where black holes tear stars or other objects apart, creating bursts of intense light.
Black Hole Meal Sets Record for Length and Size

Observing clusters of stars to find intermediate mass black holes, and modeling how they shape their environments.
A Middleweight Black Hole is Hiding at the Center of a Giant Star Cluster

Hunting for and characterizing stellar mass black holes, which can include information about their birth process and evolution.
NASA's Chandra Adds to Black Hole Birth Announcement

The Varieties of Black Holes
Black holes come in three categories:

Stellar Mass Black Holes are born from the death of stars much more massive than the Sun. When some of these stars run out of the nuclear fuel that makes them shine, their cores collapse into black holes under their own gravity. Other stellar mass black holes form from the collision of neutron stars, such as the ones first detected by LIGO and Virgo in 2017. These are probably the most common black holes in the cosmos, but are hard to detect unless they have an ordinary star for a companion. When that happens, the black hole can strip material from the star, causing the gas to heat up and glow brightly in X-rays.

Supermassive Black Holes are the monsters of the universe, living at the centers of nearly every galaxy. They range in mass from 100,000 to billions of times the mass of the Sun, far too massive to be born from a single star. The Milky Way's black hole is about 4 million times the Sun's mass, putting it in the middle of the pack. In the form of quasars and other “active” galaxies, these black holes can shine brightly enough to be seen from billions of light-years away. Understanding when these black holes formed and how they grow is a major area of research. Center for Astrophysics | Harvard & Smithsonian scientists are part of the Event Horizon Telescope (EHT) collaboration, which captured the first-ever image of the black hole: the supermassive black hole at the center of the galaxy M87.

Intermediate Mass Black Holes are the most mysterious, since we've hardly seen any of them yet. They weigh 100 to 10,000 times the mass of the Sun, putting them between stellar and supermassive black holes. We don't know exactly how many of these are, and like supermassive black holes, we don't fully understand how they're born or grow. However, studying them could tell us a lot about how the most supermassive black holes came to be.

 

Black holes can seem bizarre and incomprehensible, but in truth they're remarkably understandable. Despite not being able to see black holes directly, we know quite a bit about them. They are …

Simple. All three black hole types can be described by just two observable quantities: their mass and how fast they spin. That's much simpler than a star, for example, which in addition to mass is a product of its unique history and evolution, including its chemical makeup. Mass and spin tell us everything we need to know about a black hole: it “forgets” everything that went into making it. Those two quantities determine how big the event horizon is, and the way gravity affects any matter falling onto the black hole.

Compact. Black holes are tiny compared to their mass. The event horizon of a black hole the mass of the Sun would be no more than 6 kilometers across, and the faster it spins, the smaller that size is. Even a supermassive black hole would fit easily inside our Solar System.

Powerful. The combination of large mass and small size results in very strong gravity. This gravity is strong enough to pull a star apart if it gets too close, producing powerful bursts of light. A supermassive black hole heats gas falling onto it to temperatures of millions of degrees, making it glow brightly enough in X-rays and other types of radiation to be seen across the universe.

Very common. From theoretical calculations based on observations, astronomers think the Milky Way might have as many as a hundred million black holes, most of which are stellar mass. And with at least one supermassive black hole in most galaxies, there could be hundreds of billions of supermassive black holes in the observable universe.

Very important. Black holes have a reputation for eating everything that comes by, but they turn out to be messy eaters. A lot of stuff that falls toward a black hole gets jetted away, thanks to the complicated churning of gas near the event horizon. These jets and outflows of gas called “winds” spread atoms throughout the galaxy, and can either boost or throttle the birth of new stars, depending on other factors. That means supermassive black holes play an important role in the life of galaxies, even far beyond the black hole's gravitational pull.

And yes, mysterious. Along with astronomers, physicists are interested in black holes because they're a laboratory for “quantum gravity”. Black holes are described by Albert Einstein's general relativity, which is our modern theory of gravity, but the other forces of nature are described by quantum physics. So far, nobody has developed a complete quantum gravity theory, but we already know black holes will be an important test of any proposed theory.
"""

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less. 
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text: 
{sample_text}

Return the text with <<<SPLIT>>> markers where you want to split.
"""

print("AI chunk generation initiated.")
print("="*50)

response = llm.invoke(prompt)
marked_text = response.content

chunks = marked_text.split("<<<SPLIT>>>")

clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:
        clean_chunks.append(cleaned)
        
print("Chunking result.")
print("="*50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f"'{chunk}'\n")