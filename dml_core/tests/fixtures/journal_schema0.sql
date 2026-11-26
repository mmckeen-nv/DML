-- Legacy DML journal layout at 48b2ba21ebc4f2b479b72062557afdf9b4398be4.
-- Commit-pinned compatibility fixture, not a claim of a versioned release.
CREATE TABLE records (bucket TEXT, key TEXT, position INTEGER, payload TEXT, PRIMARY KEY(bucket,key));
CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER, metadata TEXT);
CREATE TABLE snapshot (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER, payload TEXT);
INSERT INTO state VALUES (1,2,'{"next_id":2}');
INSERT INTO records VALUES ('items','1',0,'{"id":1,"text":"legacy evidence","embedding":[1,0],"timestamp":1.0,"salience":0.5,"fidelity":1.0,"level":0,"meta":{"source":"fixture"},"summary_of":[]}');
