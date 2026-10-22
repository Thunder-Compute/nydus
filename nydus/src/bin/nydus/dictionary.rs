use std::path::PathBuf;

use clap::{Args, Subcommand};
use nydus::build::dictionary::Dictionary;
use nydus_error::Result;

#[derive(Debug, Clone, Args)]
pub struct DictionaryCommand {
    #[command(subcommand)]
    command: Operation,
}

#[derive(Debug, Clone, Subcommand)]
enum Operation {
    /// Create an empty dictionary directory for one organization.
    Init { path: PathBuf },
    /// Atomically add each already-published full blob to the dictionary.
    Add {
        #[arg(long)]
        dictionary: PathBuf,
        #[arg(required = true)]
        blobs: Vec<PathBuf>,
    },
    /// Merge another dictionary from the same organization; an empty destination becomes a snapshot.
    Merge {
        #[arg(long)]
        dictionary: PathBuf,
        source: PathBuf,
    },
    /// List the dictionary's source blobs as JSON.
    Info { path: PathBuf },
}

impl DictionaryCommand {
    pub fn execute(&self) -> Result<()> {
        match &self.command {
            Operation::Init { path } => {
                Dictionary::open(path)?;
            }
            Operation::Add { dictionary, blobs } => {
                let dictionary = Dictionary::open(dictionary)?;
                for blob in blobs {
                    println!("{}", dictionary.add_blob(blob)?.to_hex());
                }
            }
            Operation::Merge { dictionary, source } => {
                std::fs::create_dir_all(dictionary)?;
                if std::fs::canonicalize(dictionary)? == std::fs::canonicalize(source)? {
                    return Err(nydus_error::Error::InvalidParameter(
                        "dictionary merge source and destination must differ".into(),
                    ));
                }
                let source = Dictionary::open_read_only(source)?;
                Dictionary::open(dictionary)?.merge(&source)?;
            }
            Operation::Info { path } => {
                let dictionary = Dictionary::open_read_only(path)?;
                println!("{}", serde_json::to_string_pretty(&dictionary.blobs()?)?);
            }
        }
        Ok(())
    }
}
