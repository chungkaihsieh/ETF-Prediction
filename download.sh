#!/bin/bash
#dl should be equal to 1
curl -L https://www.dropbox.com/sh/mjgr2trf9qe2egr/AAAf9RebRDDSdD_XRRMo_tGxa?dl=1 > download.zip
unzip download.zip
rm -r download.zip
