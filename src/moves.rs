use std::fmt::Display;

use crate::errors::CubeError;

#[derive(PartialEq, Clone, Copy, Debug, PartialOrd)]
#[repr(u8)]
pub enum MetricKind {
    QuarterTurnMetric = 12,
    HalfTurnMetric = 18,
}

impl Display for MetricKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricKind::QuarterTurnMetric => write!(f, "QuarterTurnMetric"),
            MetricKind::HalfTurnMetric => write!(f, "HalfTurnMetric"),
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(u8)]
pub enum Turn {
    L,  // Clockwise Left turn
    R,  // Clockwise Right turn
    F,  // Clockwise Front turn
    B,  // Clockwise Back turn
    U,  // Clockwise Up turn
    D,  // Clockwise Down turn
    L_, // Anti-Clockwise Left turn
    R_, // Anti-Clockwise Right turn
    F_, // Anti-Clockwise Front turn
    B_, // Anti-Clockwise Back turn
    U_, // Anti-Clockwise Up turn
    D_, // Anti-lockwise Down turn
    L2, // Half Left turn
    R2, // Half Right turn
    F2, // Half Front turn
    B2, // Half Back turn
    U2, // Half Up turn
    D2, // Half Down turn
}

impl Display for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Turn::L => write!(f, "L "),
            Turn::R => write!(f, "R "),
            Turn::F => write!(f, "F "),
            Turn::B => write!(f, "B "),
            Turn::U => write!(f, "U "),
            Turn::D => write!(f, "D "),
            Turn::L_ => write!(f, "L_ "),
            Turn::R_ => write!(f, "R_ "),
            Turn::F_ => write!(f, "F_ "),
            Turn::B_ => write!(f, "B_ "),
            Turn::U_ => write!(f, "U_ "),
            Turn::D_ => write!(f, "D_ "),
            Turn::L2 => write!(f, "L2 "),
            Turn::R2 => write!(f, "R2 "),
            Turn::F2 => write!(f, "F2 "),
            Turn::B2 => write!(f, "B2 "),
            Turn::U2 => write!(f, "U2 "),
            Turn::D2 => write!(f, "D2 "),
        }
    }
}

impl Turn {
    pub fn from_u8(value: u8) -> Result<Self, CubeError> {
        match value {
            0u8 => Ok(Turn::L),
            1u8 => Ok(Turn::R),
            2u8 => Ok(Turn::F),
            3u8 => Ok(Turn::B),
            4u8 => Ok(Turn::U),
            5u8 => Ok(Turn::D),
            6u8 => Ok(Turn::L_),
            7u8 => Ok(Turn::R_),
            8u8 => Ok(Turn::F_),
            9u8 => Ok(Turn::B_),
            10u8 => Ok(Turn::U_),
            11u8 => Ok(Turn::D_),
            12u8 => Ok(Turn::L2),
            13u8 => Ok(Turn::R2),
            14u8 => Ok(Turn::F2),
            15u8 => Ok(Turn::B2),
            16u8 => Ok(Turn::U2),
            17u8 => Ok(Turn::D2),
            _ => Err(CubeError::InvalidTurn(value, 0)), // code never reaches here, check done in turn
        }
    }
}
