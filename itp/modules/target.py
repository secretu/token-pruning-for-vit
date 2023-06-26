sing_octo_target = \
"""
target:
  service: aisc
  name: msroctovc

environment:
  image: amlt-sing/pytorch-1.8.0
  # sh does not have "source", so use ". ./xxx.sh" here.
  setup:
  - . ./itp/setup.sh
"""

sing_research_target = \
"""
target:
  service: aisc
  name: msrresrchvc

environment:
  image: amlt-sing/pytorch-1.8.0
  # sh does not have "source", so use ". ./xxx.sh" here.
  setup:
  - . ./itp/setup.sh
"""

itp_rr1_target = \
"""
target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: itplabrr1cl1 # choices:
                     # MSR-Lab-RR1-V100-32GB: itplabrr1cl1, 
                     # ads: v100-8x-eus-1
                          
  vc: resrchvc

environment:
  image: v-xudongwang/pytorch:taoky
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
  setup:
  - . ./itp/setup.sh
"""

itp_p100_target = \
"""
target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: itpeusp100cl
  vc: resrchvc

environment:
  image: v-xudongwang/pytorch:taoky
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
  setup:
  - . ./itp/setup.sh
"""
itp_ads_target = \
"""
target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: v100-8x-eus-1 # choices:
                      # MSR-Lab-RR1-V100-32GB: itplabrr1cl1, 
                      # ads: v100-8x-eus-1
                          
  vc: ads

environment:
  image: v-xudongwang/pytorch:taoky
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
  setup:
  - . ./itp/setup.sh
"""
itp_adsp40_target = \
"""
target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: itp-p40-eus # choices:
                      # MSR-Lab-RR1-V100-32GB: itplabrr1cl1, 
                      # ads: v100-8x-eus-1
                          
  vc: ads

environment:
  image: v-xudongwang/pytorch:taoky
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
  setup:
  - . ./itp/setup.sh
"""

itp_ads_gpt3_target = \
"""
target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: v100-8x-wus2 # choices:
                      # MSR-Lab-RR1-V100-32GB: itplabrr1cl1, 
                      # ads: v100-8x-eus-1
                          
  vc: Ads-GPT3

environment:
  image: v-xudongwang/pytorch:taoky
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
  setup:
  - . ./itp/setup.sh
"""

itp_ads_a100_target = \
"""
target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: a100-8x-wus2
  vc: ads

environment:
  image: v-xudongwang/pytorch:cddlyf_a100
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
  setup:
  - . ./itp/setup.sh
"""

target_dict = dict(
    sing_octo=sing_octo_target,
    sing_research=sing_research_target,
    itp_rr1=itp_rr1_target,
    itp_p100=itp_p100_target,
    itp_ads_v100=itp_ads_target,
    itp_ads_gpt3_v100=itp_ads_gpt3_target,
    itp_ads_a100=itp_ads_a100_target,
    itp_ads_p40=itp_adsp40_target
)
