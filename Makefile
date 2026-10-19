main:
	snakemake --profile profiles/local

check:
	snakemake --profile profiles/local -n

clean:
	rm -rf .snakemake/ build/ notebooks/generated/
