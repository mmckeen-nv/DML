# DPM - DML Personality/Continuity Plugin

Purpose:
- Build a toggleable plugin layer on top of DML for thread continuity, relationship continuity, and weighted preference modeling.
- Keep core DML substrate changes minimal and additive.

Working rules:
- Work in this project directory first.
- Keep git state clean and intentional.
- Treat durable DML as the promotion target, not the scratchpad.

Initial scope:
- continuity metadata contract
- replay/checkpoint format
- thread/project/global retrieval policy
- weighted preference/value graph
- plugin enable/disable design
- validation bundle
